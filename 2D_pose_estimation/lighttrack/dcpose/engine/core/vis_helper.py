#!/usr/bin/python
# -*- coding:utf8 -*-
import cv2
from random import random
import os.path as osp

from utils.utils_image import read_image, save_image
from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from datasets.zoo.infant.pose_skeleton import PoseTrack_Official_Keypoint_Ordering, PoseTrack_Keypoint_Pairs, PoseTrack_COCO_Keypoint_Pair, PoseTrack_COCO_Keypoint_Ordering
from utils.utils_color import COLOR_DICT


def draw_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir, vis_skeleton=True, vis_bbox=True):
    """
    :param batch_image_list:  batch image path
    :param batch_joints_list:   joints coordinates in image Coordinate reference system
    :batch_bbox_list: xyxy
    :param save_dir:
    :return: No return
    """

    skeleton_image_save_folder = osp.join(save_dir, "skeleton")
    bbox_image_save_folder = osp.join(save_dir, "bbox")
    together_save_folder = osp.join(save_dir, "SkeletonAndBbox")
    heatmap_folder = osp.join(save_dir, "heatmap")
    if vis_skeleton and vis_bbox:
        save_folder = together_save_folder
    else:
        save_folder = skeleton_image_save_folder
        if vis_bbox:
            save_folder = bbox_image_save_folder

    batch_final_coords = batch_joints_list

    for index, image_path in enumerate(batch_image_list):
        final_coords = batch_final_coords[index]
        #final_coords = coco2posetrack_ord_infer(final_coords)
        bbox = batch_bbox_list[index]

        image_name = osp.join(*image_path.split('/')[-3:])
        # image_name = image_path[image_path.index("frames") + len("frames") + 1:]

        vis_image_save_path = osp.join(save_folder, image_name)
        vis_heatmap_save_path = osp.join(heatmap_folder, image_name)
        batch_image = []

        if False :#osp.exists(vis_image_save_path):
            processed_image = read_image(vis_image_save_path)

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image
            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)
        else:
            image_data = read_image(image_path)
            batch_image.append(image_data)
            processed_image = image_data.copy()

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image

            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)

        # if debug_heatmap_pred:
        #     save_batch_heatmaps(
        #         torch.cat(batch_image, dim=0), output, '{}_hm_pred.jpg'.format(prefix)
        #     )

def add_bbox_in_image(image, bbox):
    """
    :param image
    :param bbox   -  xyxy
    """

    color = (random() * 255, random() * 255, random() * 255)

    x1, y1, x2, y2 = map(int, bbox)
    image_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=6)
    return image_with_bbox


def add_poseTrack_joint_connection_to_image(img_demo, joints, sure_threshold=0.8, flag_only_draw_sure=False, ):
    for joint_pair in PoseTrack_COCO_Keypoint_Pair:
        ind_1 = PoseTrack_COCO_Keypoint_Ordering.index(joint_pair[0])
        ind_2 = PoseTrack_COCO_Keypoint_Ordering.index(joint_pair[1])

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1 <= 5: continue
        if x2 <= 5 and y2 <= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > sure_threshold and sure2 > sure_threshold:
            # if sure1 > 0.8 and sure2 > 0.8:
            # cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=6)
    return img_demo


def circle_vis_point(img, joints):
    for joint in joints:
        x, y, c = [int(i) for i in joint]
        cv2.circle(img, (x, y), 3, (255, 255, 255), thickness=3)

    return img


from datasets.process.heatmaps_process import get_max_preds

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    #print(nmaps)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_debug_images(batch_image_list, target, joints_pred, output,
                      save_dir, debug_heatmap_gt=True, debug_heatmap_pred=True):
    if not config.DEBUG.DEBUG:
        return

    

