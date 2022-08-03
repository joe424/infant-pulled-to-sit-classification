#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os.path as osp
import torch
import copy
import random
import cv2
from pycocotools.coco import COCO
import logging
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored

from .posetrack_utils import video2filenames, evaluate_simple
from utils.utils_json import read_json_from_file, write_json_to_file
from utils.utils_bbox import box2cs
from utils.utils_image import read_image
from utils.utils_folder import create_folder
from utils.utils_registry import DATASET_REGISTRY
from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct

from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


@DATASET_REGISTRY.register()
class PoseTrackO(VideoDataset):
    """
        PoseTrackO
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrackO, self).__init__(cfg, phase, **kwargs)

        self.train = True if phase == TRAIN_PHASE else False
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
                                      dtype=np.float32).reshape((self.num_joints, 1))
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.is_fullset = cfg.DATASET.IS_FULLSET
        self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        self.transform = build_transforms(cfg, phase)

        self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        self.previous_distance = cfg.PREVIOUS_DISTANCE
        self.next_distance = cfg.NEXT_DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE

        if self.phase != TRAIN_PHASE:
            self.img_dir = cfg.DATASET.TEST_IMG_DIR
            temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
            self.annotation_dir = temp_subCfgNode.ANNOT_DIR
        if self.is_fullset:
            self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val.json'))
        else:
            self.coco = COCO(osp.join(self.json_dir, 'posetrack_sub_train.json' if self.is_train else 'posetrack_sub_val.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.data = self._list_data()

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

    def __getitem__(self, item_index):
        data_item = copy.deepcopy(self.data[item_index])
        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            return self._get_spatiotemporal_window(data_item)

    def _get_spatiotemporal_window(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        if self.distance_whole_otherwise_segment:
            farthest_distance = self.distance
            prev_delta_range = range(1, min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance))
            next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
                                            farthest_distance))
        else:
            prev_delta_range = range(1, min(current_idx + 1 if is_posetrack18 else current_idx, self.previous_distance))
            next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
                                            self.next_distance))

        prev_delta_range = list(prev_delta_range)
        next_delta_range = list(next_delta_range)
        # setting deltas

        if len(prev_delta_range) == 0:
            prev_delta = 0
            margin_left = 1
        else:
            if self.random_aux_frame:
                prev_delta = random.choice(prev_delta_range)
            else:
                prev_delta = prev_delta_range[-1]
            margin_left = prev_delta

        if len(next_delta_range) == 0:
            next_delta = 0
            margin_right = 1
        else:
            if self.random_aux_frame:
                next_delta = random.choice(next_delta_range)
            else:
                next_delta = next_delta_range[-1]

            margin_right = next_delta

        prev_idx = current_idx - prev_delta
        next_idx = current_idx + next_delta

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')

        # checking for files existence
        if not osp.exists(prev_image_file):
            error_msg = "Can not find image :{}".format(prev_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        if not osp.exists(next_image_file):
            error_msg = "Can not find image :{}".format(next_image_file)
            self.logger.error(error_msg)
            raise Exception(error_msg)

        data_numpy_prev = read_image(prev_image_file)
        data_numpy_next = read_image(next_image_file)

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev = cv2.cvtColor(data_numpy_prev, cv2.COLOR_BGR2RGB)
            data_numpy_next = cv2.cvtColor(data_numpy_next, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            self.logger.error('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev is None:
            self.logger.error('=> PREV SUP: fail to read {}'.format(prev_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_file))
        if data_numpy_next is None:
            self.logger.error('=> NEXT SUP: fail to read {}'.format(next_image_file))
            raise ValueError('NEXT SUP: Fail to read {}'.format(next_image_file))

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        ### visibility of a person (one-hot) [dont_know_order, foreground, background]
        depth_label = data_item['depth_order']
        vis_input = torch.zeros((4)).float()
        vis_input.scatter_(0, torch.tensor(depth_label).long(), 1.0)
        # avg_vis = torch.mean(torch.tensor(joints_vis))
        # avg_vis = 1 if avg_vis > 0.5 else 0
        # vis_input = torch.zeros((3))
        # if random.uniform(0, 1) < 0.3:
        #     vis_input = torch.tensor([1,0,0], dtype=torch.float)
        # else:
        #     vis_input[1] = avg_vis
        #     vis_input[2] = 1-avg_vis

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

            # jitter bounding box 
            self.bbx_jitter = True
            if self.bbx_jitter and random.random() <= 0.5:
                x_rand_jit = random.randint(-10,10)
                y_rand_jit = random.randint(-10,10)
                center[0] = np.clip(center[0] + x_rand_jit, 0, data_numpy.shape[1]).astype(int).item()
                center[1] = np.clip(center[1] + y_rand_jit, 0, data_numpy.shape[0]).astype(int).item()
                for i in range(self.num_joints):
                    if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                        joints[i, 0] = np.clip(joints[i, 0] + x_rand_jit, 0, data_numpy.shape[1]).astype(int).item()
                        joints[i, 1] = np.clip(joints[i, 1] + y_rand_jit, 0, data_numpy.shape[0]).astype(int).item()

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)



        # vis_label = torch.tensor(1) if avg_vis == 1 else torch.tensor(2)

        meta = {
            'image': image_file_path,
            'prev_sup_image': prev_image_file,
            'next_sup_image': next_image_file,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': vis_input,
            # 'joints_vis_label': vis_label,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'margin_left': margin_left,
            'margin_right': margin_right,
            'depth_order': data_item['depth_order'],
        }

        return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta

    def _get_single_frame(self, data_item):
        raise NotImplementedError

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()
        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            xyxy_name_list = ["x1", "y1", "x2", "y2"]
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 > x1 and y2 > y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    obj['xyxy_bbox'] = {coord: val for coord, val in zip(xyxy_name_list, [x1, y1, x2, y2])}
                    valid_objs.append(obj)
            objs = valid_objs

            # define foreground/background over occlusion among people
            is_group_crowded_person = False
            people_occlusion_thres = 0.4
            inter_occlusion_matrix = torch.zeros((len(objs), len(objs)))
            for i in range(len(objs)):
                for j in range(i,len(objs)):
                    if i == j:
                        continue
                    pose1 = objs[i]['keypoints']
                    pose2 = objs[j]['keypoints']
                    bboxes1_dict = objs[i]['xyxy_bbox']
                    bboxes2_dict = objs[j]['xyxy_bbox']
                    iou = self.get_iou(bboxes1_dict, bboxes2_dict) 
                    if self.check_inter_occlusion(pose1, pose2, bboxes1_dict, bboxes2_dict) == True \
                    or iou > people_occlusion_thres:
                        
                        # Flag for variable data unit. If True, it will pack two persons annotations instead of one.
                        is_group_crowded_person = True 
                        
                        if self.check_foreground(objs[i]['keypoints'], objs[j]['keypoints']):
                            # depth_order = 0 (default) when there is no occlusion among people
                            # depth_order = 1 when foreground, 
                            # depth_order = 2 when middle, 
                            # depth_order = 3 when background,
                            inter_occlusion_matrix[i,j] = 1
                            inter_occlusion_matrix[j,i] = 3
                        else:
                            inter_occlusion_matrix[i,j] = 3
                            inter_occlusion_matrix[j,i] = 1

            rec = []
            for obj_index, obj in enumerate(objs):
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)

                if not is_group_crowded_person:
                    depth_order = 0
                else:
                    order = torch.unique(inter_occlusion_matrix[i])
                    depth_order = int(torch.max(inter_occlusion_matrix[obj_index]).item()) if len(order) == 2 else 2
                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                    'depth_order': depth_order,
                })
            gt_db.extend(rec)
        return gt_db

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })
        logger.info('=> Total boxes: {}'.format(len(all_boxes)))
        logger.info('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, num_boxes))

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_tracks = []
        cc = 0

        # print(img_path)
        for key in img_path:
            temp = key.split('/')

            # video_name = osp.dirname(key)
            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_fullset = self.is_fullset
        is_posetrack18 = self.is_posetrack18
        if is_fullset:
            pass
        else:
            annot_dir = annot_dir.replace("val", "sub_val")
        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            cur_length = L['images/' + vid]

            temp_kps_map = {}
            temp_track_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
            #### including empty frames
            nnz_counter = 0
            next_track_id = 0

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length
            # start id ~ finish id
            for current_frame_id in range(sid, fid):
                frame_num = current_frame_id
                if not current_frame_id in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                    else:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                    kps = []
                    tracks = []
                    bboxs = []

                else:

                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bboxs = temp_box_map[frame_num]
                    tracks = [track_id for track_id in range(len(kps))]
                    # tracks = [1] * len(kps)

                ### creating a data element
                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        logger.info("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
        AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False)[0]

        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

    def check_foreground(self, pose1, pose2):
        """
        Determine foreground/background of two occluded persons.
        Return True, if pose1_score > pose2_score. Otherwise return False.
        ---------
        """
        pose1 = torch.tensor(pose1).view(self.num_joints, -1)
        pose2 = torch.tensor(pose2).view(self.num_joints, -1)
        pose1occpose2_score, pose2occpose1_score =0, 0
        for joint1 in pose1:
            if joint1[2] > 0:
                for joint2 in pose2:
                    if joint2[2] > 0:
                        continue
                    else:
                        if torch.sum((joint2 - joint1)**2) < 400: # (50 pixl)**2
                            pose1occpose2_score += 1
        for joint2 in pose2:
            if joint2[2] > 0:
                for joint1 in pose1:
                    if joint1[2] > 0:
                        continue
                    else:
                        if torch.sum((joint2 - joint1)**2) < 400: # (50 pixl)**2
                            pose2occpose1_score += 1
        return True if pose1occpose2_score >  pose2occpose1_score else False
    def check_inter_occlusion(self, pose1, pose2, bb1, bb2, thresh=3):
        assert bb1["x1"] < bb1["x2"]
        assert bb1["y1"] < bb1["y2"]
        assert bb2["x1"] < bb2["x2"]
        assert bb2["y1"] < bb2["y2"]
        joint_indexs = list(range(len(pose1)))
        
        # pose1 to bb2
        joint_in_bbox_cnt1 = 0
        for ith_joint_index in joint_indexs[::3]:
            joint = pose1[ith_joint_index:ith_joint_index+3]
            if joint[2] == 0:
                continue
            if bb2["x1"] <= joint[0] and joint[0] <= bb2["x2"] and\
            bb2["y1"] <= joint[1] and joint[1] <= bb2["y2"]:
                joint_in_bbox_cnt1 += 1
        
        # pose2 to bb1     
        joint_in_bbox_cnt2 = 0
        for ith_joint_index in joint_indexs[::3]:
            joint = pose2[ith_joint_index:ith_joint_index+3]
            if joint[2] == 0:
                continue
            if bb1["x1"] <= joint[0] and joint[0] <= bb1["x2"] and\
            bb1["y1"] <= joint[1] and joint[1] <= bb1["y2"]:
                joint_in_bbox_cnt2 += 1
        
        return True if joint_in_bbox_cnt1 >= thresh or joint_in_bbox_cnt2 >= thresh else False
        
    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1["x1"] < bb1["x2"]
        assert bb1["y1"] < bb1["y2"]
        assert bb2["x1"] < bb2["x2"]
        assert bb2["y1"] < bb2["y2"]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1["x1"], bb2["x1"])
        y_top = max(bb1["y1"], bb2["y1"])
        x_right = min(bb1["x2"], bb2["x2"])
        y_bottom = min(bb1["y2"], bb2["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
        bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
    # def jitter_bbx(self,bbx, keypoints, jitter_range):
        
        