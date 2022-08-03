'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    April 23rd, 2019
    LightTrack: A Generic Framework for Online Top-Down Human Pose Tracking
    Demo on videos using YOLOv3 detector and Mobilenetv1-Deconv.
'''
import time
import argparse
import glob
import copy
from tqdm import tqdm
#path
import _init_paths
import sys

# import vision essentials
import cv2
import numpy as np

import os
import os.path as osp


# object detector
from utils import *
from detector.detector_yolov3 import *

# Infant skeleton (pose) estimator
from dcpose.tools.inference import inference_PE

# import GCN utils
sys.path.append(os.path.abspath("./graph/"))
from graph import visualize_pose_matching
from graph  .visualize_pose_matching import *

# import my own utils
from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *
from utils_extract_infant_skeletons import extract_infant_skeleton
import warnings

flag_visualize = False
flag_nms = False #Default is False, unless you know what you are doing

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def initialize_parameters():
    global video_name, img_id

    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale, pose_matching_threshold
    keyframe_interval =  4000 # choice examples: [2, 3, 5, 8, 10, 20, 40, 100, ....]
    enlarge_scale = 0.2 # how much to enlarge the bbox before pose estimation
    pose_matching_threshold = 0.5


    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    total_time_POSE = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0
    return


def light_track(image_folder, 
                output_json_path, visualize_folder, output_video_path, output_pkl_dir):

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    ''' 1. statistics: get total time for lighttrack processing'''
    st_time_total = time.time()

    # process the frames sequentially
    keypoints_list = []
    bbox_dets_list = []
    frame_prev = -1
    frame_cur = 0
    img_id = -1
    next_id = 0
    bbox_dets_list_list = []
    keypoints_list_list = []

    flag_mandatory_keyframe = False

    img_paths = get_immediate_childfile_paths(image_folder)
    num_imgs = len(img_paths)
    total_num_FRAMES = num_imgs

    while img_id < num_imgs-1:
        img_id += 1
        img_path = img_paths[img_id]
    
        print("\r\t[{}/{} frames] [image_id:{}], ".format(img_id, num_imgs,osp.basename(img_path), img_id), end="")

        frame_cur = img_id
        if (frame_cur == frame_prev):
            frame_prev -= 1

        ''' KEYFRAME: loading results from other modules '''
        if is_keyframe(img_id, keyframe_interval) or flag_mandatory_keyframe:
            flag_mandatory_keyframe = False
            bbox_dets_list = []  # keyframe: start from empty
            keypoints_list = []  # keyframe: start from empty

            # perform detection at keyframes
            st_time_detection = time.time()
            human_candidates = inference_yolov3(img_path)

            end_time_detection = time.time()
            total_time_DET += (end_time_detection - st_time_detection)

            num_dets = len(human_candidates)
#             print("[{} detections], ".format(num_dets), end="")

            # if nothing detected at keyframe, regard next frame as keyframe because there is nothing to track
            if num_dets <= 0:
                # add empty result
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":  0,
                                 "track_id": None,
                                 "imgpath": img_path,
                                 "bbox": [0, 0, 2, 2]}
                bbox_dets_list.append(bbox_det_dict)

                keypoints_dict = {"img_id":img_id,
                                  "det_id": 0,
                                  "track_id": None,
                                  "imgpath": img_path,
                                  "keypoints": []}
                keypoints_list.append(keypoints_dict)

                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)

                flag_mandatory_keyframe = True
                continue

            ''' 2. statistics: get total number of detected persons '''
            total_num_PERSONS += num_dets

            if img_id > 0:   # First frame does not have previous frame
                # pool_size_acumu = 0
                # pool_size_acumu = pool_size_acumu+1 
                pool_size = 8 if img_id >=8 else img_id
                bbox_list_prev_frame = copy.deepcopy(bbox_dets_list_list[img_id - pool_size:])
                # print("len(bbox_list_prev_frame)", len(bbox_list_prev_frame))
                keypoints_list_prev_frame = copy.deepcopy(keypoints_list_list[img_id - pool_size:])

            # For each candidate, perform pose estimation and data association based on Spatial Consistency (SC)
            for det_id in range(num_dets):
                # obtain bbox position and track id
                bbox_det = human_candidates[det_id]

                # enlarge bbox by 20% with same center position
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                # Keyframe: use provided bbox
                if bbox_invalid(bbox_det):
                    track_id = None # this id means null
                    keypoints = []
                    bbox_det = [0, 0, 2 ,2]
                    # update current frame bbox
                    bbox_det_dict = {"img_id":img_id,
                                     "det_id":det_id,
                                     "track_id": track_id,
                                     "imgpath": img_path,
                                     "bbox":bbox_det}
                    bbox_dets_list.append(bbox_det_dict)
                    # update current frame keypoints
                    keypoints_dict = {"img_id":img_id,
                                      "det_id":det_id,
                                      "track_id": track_id,
                                      "imgpath": img_path,
                                      "keypoints":keypoints}
                    keypoints_list.append(keypoints_dict)
                    continue

                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "imgpath": img_path,
                                 "bbox":bbox_det}

                # obtain keypoints for each bbox position in the keyframe
                st_time_pose = time.time()
                # print("inference_keypoints_keyframe")
                # bbx = torch.tensor(bbx_nms(bbox_det_dict)).view(-1,6)
                keypoints = inference_PE(bbox_det_dict["imgpath"],None, None,  bbox_det_dict["bbox"]).reshape(-1)
                keypoints = keypoints.tolist()
                end_time_pose = time.time()
                total_time_POSE += (end_time_pose - st_time_pose)

                if img_id == 0:   # First frame, all ids are assigned automatically
                    track_id = next_id
                    next_id += 1
                else:
                    track_id = -1
                    for i in range(pool_size):
                        track_id, match_index = get_track_id_SpatialConsistency(bbox_det, bbox_list_prev_frame[pool_size-1-i])

                        if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                            # print("get_track_id_SpatialConsistency Success at track_id:", track_id )
                            del bbox_list_prev_frame[pool_size-1-i][match_index]
                            del keypoints_list_prev_frame[pool_size-1-i][match_index]
                            break
#                     print("[track SpatialConsistency:{} times], ".format(i), end="")
                    
                # update current frame bbox
                bbox_det_dict = {"img_id":img_id,
                                 "det_id":det_id,
                                 "track_id":track_id,
                                 "imgpath": img_path,
                                 "bbox":bbox_det}
                bbox_dets_list.append(bbox_det_dict)

                # update current frame keypoints
                keypoints_dict = {"img_id":img_id,
                                  "det_id":det_id,
                                  "track_id":track_id,
                                  "imgpath": img_path,
                                  "keypoints":keypoints}
                keypoints_list.append(keypoints_dict)

            # For candidate that is not assopciated yet, perform data association based on Pose Similarity (SGCN)
            for det_id in range(num_dets):
                bbox_det_dict = bbox_dets_list[det_id]
                keypoints_dict = keypoints_list[det_id]
                assert(det_id == bbox_det_dict["det_id"])
                assert(det_id == keypoints_dict["det_id"])

                if bbox_det_dict["track_id"] == -1:    # this id means matching not found yet
                    for i in range(pool_size):
                        track_id, match_index = get_track_id_SGCN(bbox_det_dict["bbox"], bbox_list_prev_frame[pool_size-1-i],
                                                                    keypoints_dict["keypoints"], keypoints_list_prev_frame[pool_size-1-i])

                        if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                            del bbox_list_prev_frame[pool_size-1-i][match_index]
                            del keypoints_list_prev_frame[pool_size-1-i][match_index]
                            bbox_det_dict["track_id"] = track_id
                            keypoints_dict["track_id"] = track_id
                            break
                    # if still can not find a match from previous frame, then assign a new id
                    if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                        bbox_det_dict["track_id"] = next_id
                        keypoints_dict["track_id"] = next_id
                        next_id += 1

            # update frame
            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)
            frame_prev = frame_cur

        else:
            ''' NOT KEYFRAME: multi-target pose tracking '''
            bbox_dets_list_next = []
            keypoints_list_next = []

            num_dets = len(keypoints_list)
            total_num_PERSONS += num_dets

            if num_dets == 0:
                flag_mandatory_keyframe = True

            for det_id in range(num_dets):
                keypoints = keypoints_list[det_id]["keypoints"]

                # for non-keyframes, the tracked target preserves its track_id
                track_id = keypoints_list[det_id]["track_id"]

                # next frame bbox
                bbox_det_next = get_bbox_from_keypoints(keypoints)
                #bbox_det_next = human_candidates[0]
                #print(bbox_det_next)
                if bbox_det_next[2] == 0 or bbox_det_next[3] == 0:
                    bbox_det_next = [0, 0, 2, 2]
                    total_num_PERSONS -= 1
                assert(bbox_det_next[2] != 0 and bbox_det_next[3] != 0) # width and height must not be zero
                bbox_det_dict_next = {"img_id":img_id,
                                     "det_id":det_id,
                                     "track_id":track_id,
                                     "imgpath": img_path,
                                     "bbox":bbox_det_next}

                # next frame keypoints
                st_time_pose = time.time()
                # bbx = torch.tensor(bbx_nms(bbox_det_dict_next)).view(-1,6)
                keypoints_next = inference_PE(bbox_det_dict_next["imgpath"], None, None, bbox_det_dict_next["bbox"]).reshape(-1) 
                keypoints_next = keypoints_next.tolist()
                # print("inference_keypoints_next", keypoints_next)
                end_time_pose = time.time()
                total_time_POSE += (end_time_pose - st_time_pose)
#                 print("[pose estimation time: {:.5f}s], ".format(end_time_pose - st_time_pose), end="")

                # check whether the target is lost
                target_lost = is_target_lost(keypoints_next)

                if target_lost is False:
                    bbox_dets_list_next.append(bbox_det_dict_next)
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "bbox":bbox_det,
                                           "track_id":track_id,
                                           "imgpath": img_path,
                                           "keypoints":keypoints_next}
                    keypoints_list_next.append(keypoints_dict_next)

                else:
                    # remove this bbox, do not register its keypoints
                    bbox_det_dict_next = {"img_id":img_id,
                                          "det_id":  det_id,
                                          "track_id": None,
                                          "imgpath": img_path,
                                          "bbox": [0, 0, 2, 2]}
                    bbox_dets_list_next.append(bbox_det_dict_next)

                    keypoints_null = 45*[0]
                    keypoints_dict_next = {"img_id":img_id,
                                           "det_id":det_id,
                                           "track_id": None,
                                           "imgpath": img_path,
                                           "keypoints": []}
                    keypoints_list_next.append(keypoints_dict_next)
#                     print("Target lost. Process this frame again as keyframe.", end="")
                    flag_mandatory_keyframe = True

                    total_num_PERSONS -= 1
                    ## Re-process this frame by treating it as a keyframe
                    if img_id not in [0]:
                        img_id -= 1
                    break

            # update frame
            if flag_mandatory_keyframe is False:
                bbox_dets_list = bbox_dets_list_next
                keypoints_list = keypoints_list_next
                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)
                frame_prev = frame_cur
            
    ''' 1. statistics: get total time for lighttrack processing'''
    end_time_total = time.time()
    total_time_ALL += (end_time_total - st_time_total)

    # convert results into openSVAI format
    print("Exporting Results in openSVAI Standard Json Format...")
    poses_standard = pose_to_standard_mot(keypoints_list_list, bbox_dets_list_list)
    #json_str = python_to_json(poses_standard)
    #print(json_str)

    # output json file
    write_json_to_file(poses_standard, output_json_path)
    print("Json Export Finished! Store in {}".format(output_json_path))

    # extract infant pose and out put output pickle.
    is_success = extract_infant_skeleton(output_json_path, output_pkl_dir)
    if is_success == False:
        low_num_skeletons.append(output_json_path)
    # visualization
    if flag_visualize is True:
#         print("\tVisualizing Pose Tracking Results...")
        create_folder(visualize_folder)
        show_all_from_standard_json(output_json_path, classes, joint_pairs, joint_names, image_folder, visualize_folder, flag_track = False)
#         print("\n\tVisualization Finished!")

        img_paths = get_immediate_childfile_paths(visualize_folder)
        avg_fps = total_num_FRAMES / total_time_ALL
        #make_video_from_images(img_paths, output_video_path, fps=avg_fps, size=None, is_color=True, format="XVID")
        make_video_from_images(img_paths, output_video_path, fps=25, size=None, is_color=True, format="XVID")


def get_track_id_SGCN(bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame, keypoints_list_prev_frame):
    assert(len(bbox_list_prev_frame) == len(keypoints_list_prev_frame))

    min_index = None
    min_matching_score = sys.maxsize
    global pose_matching_threshold
    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1

    for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        # check the pose matching score
        keypoints_dict = keypoints_list_prev_frame[det_index]
        keypoints_prev_frame = keypoints_dict["keypoints"]
        # print("keypoints_cur_frame", keypoints_cur_frame)
        # print("keypoints_prev_frame", keypoints_prev_frame)
        if keypoints_prev_frame == []:
            keypoints_prev_frame = 45 * [0]
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = bbox_list_prev_frame[min_index]["track_id"]
        return track_id, min_index


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.55
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)

        ###
        boxA_center = [(boxA[0] + boxA[2])/2, (boxA[1] + boxA[3])/2]
        boxB_center = [(boxB[0] + boxB[2])/2, (boxB[1] + boxB[3])/2]
        if boxA_center[0] - boxB_center[0]**2 + boxA_center[1] - boxB_center[1]**2  > 900:
            continue
        ###

        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    lft_eye_idx,rht_eye_idx  = 1,2
    lft_ear_idx, rht_ear_idx = 3,4
    keypoints_A_numpy = np.array(keypoints_A).reshape(-1, 3)
    eye_mid = ((keypoints_A_numpy[lft_eye_idx]+ keypoints_A_numpy[rht_eye_idx]) /2)
    ear_mid = ((keypoints_A_numpy[lft_ear_idx] + keypoints_A_numpy[rht_ear_idx]) /2)
    keypoints_A_ = np.vstack((keypoints_A_numpy[0], eye_mid, ear_mid))
    keypoints_A = np.concatenate((keypoints_A_, keypoints_A_numpy[5:])).reshape(-1).tolist()
    
    
    keypoints_B_numpy = np.array(keypoints_B).reshape(-1, 3)
    eye_mid = ((keypoints_B_numpy[lft_eye_idx]+ keypoints_B_numpy[rht_eye_idx]) /2)
    ear_mid = ((keypoints_B_numpy[lft_ear_idx] + keypoints_B_numpy[rht_ear_idx]) /2)
    keypoints_B_ = np.vstack((keypoints_B_numpy[0], eye_mid, ear_mid))
    keypoints_B = np.concatenate((keypoints_B_, keypoints_B_numpy[5:])).reshape(-1).tolist()

    if keypoints_A == [] or keypoints_B == []:
        print("\r\tgraph not correctly generated!", end="")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("\r\tgraph not correctly generated!", end="")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("\r\tgraph not correctly generated!", end="")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("\r\tgraph not correctly generated!", end="")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist



def get_iou_score(bbox_gt, bbox_det):
    boxA = xywh_to_x1y1x2y2(bbox_gt)
    boxB = xywh_to_x1y1x2y2(bbox_det)

    iou_score = iou(boxA, boxB)
    #print("iou_score: ", iou_score)
    return iou_score


def is_target_lost(keypoints, method="max_average"):
    num_keypoints = int(len(keypoints) / 3.0)
    if method == "average":
        # pure average
        score = 0
        for i in range(num_keypoints):
            score += keypoints[3*i + 2]
        score /= num_keypoints*1.0

    elif method == "max_average":
        score_list = keypoints[2::3]
        score_list_sorted = sorted(score_list)
        top_N = 4
        assert(top_N < num_keypoints)
        top_scores = [score_list_sorted[-i] for i in range(1, top_N+1)]
        score = sum(top_scores)/top_N
    if score < 0.6:
        return True
    else:
        return False


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox_from_keypoints(keypoints_python_data):
    if keypoints_python_data == [] or keypoints_python_data == 45*[0]:
        return [0, 0, 2, 2]

    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]
        if vis != 0 and vis!= 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = enlarge_scale # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


def enlarge_bbox(bbox, scale):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


# def bbx_nms(test_data):
#     cls_dets = test_data["bbox"]
#     # nms on the bboxes
#     if flag_nms is True:
#         cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
#         test_data = torch.tensor(test_data)[keep]
#         if len(keep) == 0:
#             return -1
#     else:
#         test_data = test_data
#     return test_data

def apply_nms(cls_dets, nms_method, nms_thresh):
    # nms and filter
    keep = np.where((cls_dets[:, 4] >= min_scores) &
                    ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
    cls_dets = cls_dets[keep]
    if len(cls_dets) > 0:
        if nms_method == 'nms':
            keep = gpu_nms(cls_dets, nms_thresh)
        elif nms_method == 'soft':
            keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
        else:
            assert False
    cls_dets = cls_dets[keep]
    return cls_dets, keep


def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False


def pose_to_standard_mot(keypoints_list_list, dets_list_list):
    openSVAI_python_data_list = []

    num_keypoints_list = len(keypoints_list_list)
    num_dets_list = len(dets_list_list)
    assert(num_keypoints_list == num_dets_list)

    for i in range(num_dets_list):

        dets_list = dets_list_list[i]
        keypoints_list = keypoints_list_list[i]

        if dets_list == []:
            continue
        img_path = dets_list[0]["imgpath"]
        img_folder_path = os.path.dirname(img_path)
        img_name =  os.path.basename(img_path)
        img_info = {"folder": img_folder_path,
                    "name": img_name,
                    "id": [int(i)]}
        openSVAI_python_data = {"image":[], "candidates":[]}
        openSVAI_python_data["image"] = img_info

        num_dets = len(dets_list)
        num_keypoints = len(keypoints_list) #number of persons, not number of keypoints for each person
        candidate_list = []

        for j in range(num_dets):
            keypoints_dict = keypoints_list[j]
            dets_dict = dets_list[j]

            img_id = keypoints_dict["img_id"]
            det_id = keypoints_dict["det_id"]
            track_id = keypoints_dict["track_id"]
            img_path = keypoints_dict["imgpath"]

            bbox_dets_data = dets_list[det_id]
            det = dets_dict["bbox"]
            if  det == [0, 0, 2, 2]:
                # do not provide keypoints
                candidate = {"det_bbox": [0, 0, 2, 2],
                             "det_score": 0}
            else:
                bbox_in_xywh = det[0:4]
                keypoints = keypoints_dict["keypoints"]

                track_score = sum(keypoints[2::3])/len(keypoints)/3.0

                candidate = {"det_bbox": bbox_in_xywh,
                             "det_score": 1,
                             "track_id": track_id,
                             "track_score": track_score,
                             "pose_keypoints_2d": keypoints}
            candidate_list.append(candidate)
        openSVAI_python_data["candidates"] = candidate_list
        openSVAI_python_data_list.append(openSVAI_python_data)
    return openSVAI_python_data_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.") # turn off this warning
    
    from functools import reduce
#     global args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_dir', '-v', type=str, dest='video_dir', default="../frames/")
    parser.add_argument('--out_dir', '-o', type=str, dest='out_dir', default="../out/")
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    initialize_parameters()
    output_json_dir = osp.abspath(osp.join(args.out_dir, "jsons/"))
    output_video_dir = osp.abspath(osp.join(args.out_dir, "videos/"))
    
    # directory of video frames
    source_vframe_dir = glob.glob(args.video_dir+"*/*")
    global low_num_skeletons 
    low_num_skeletons = []
    
    print("########## Extracting skeletons from images ##########")
    # for root, dirs, files in os.walk(source_vframe_dir):
    # optional 
    skip = True
    actions = ['supine', 'prone', 'pull']  
    progress_bar = 10
    print('2D pose estimation: ', end='', flush=True, file=sys.stderr)
    for i, video_path in enumerate(source_vframe_dir):
        print("[{:4d}/{:4d} videos] Current video_path: {}".format(i,len(source_vframe_dir),video_path), end="")
        # Source path
        source_image_folder = video_path + "/"

        # Output path
        v_dir, video_name = osp.split(video_path)
        output_visualize_folder = osp.abspath(osp.join(args.out_dir, "visualize/",  video_name))
        output_json_path = osp.join(output_json_dir, video_name+".json")
        output_video_path = osp.join(output_video_dir, video_name+"_out.mp4")
        for act in actions:
            if act in output_json_path.lower():
                action = act
                break
        output_pkl_dir = osp.join(args.out_dir, "extracted_skeletons", action)
        output_pkl_file = osp.join(output_pkl_dir, osp.splitext(osp.basename(output_json_path))[0])
        # if osp.exists(output_json_path):
        #     print("skeletons is predicted and stored in {}.".format(output_json_path))
        #     continue
        if osp.exists(output_json_path):
#             print("\r[{:4d}/{:4d} videos] [Extracted] Extracted infant skeletons is stored in {}.".format(i,len(source_vframe_dir), output_pkl_file))
            continue
#         print()
        create_folder(output_visualize_folder)
        create_folder(output_video_dir)
        create_folder(output_json_dir)
        create_folder(output_pkl_dir)
        # Including 1.object detection 2.pose estimation 3.tracking ID
        light_track(source_image_folder, 
                    output_json_path, output_visualize_folder, output_video_path, output_pkl_dir)

        print("\tFinished video {}".format(output_video_path))
        ''' Display statistics '''
#         print("\ttotal_time_ALL: {:.2f}s".format(total_time_ALL))
#         print("\ttotal_time_DET: {:.2f}s".format(total_time_DET))
#         print("\ttotal_time_POSE: {:.2f}s".format(total_time_POSE))
#         print("\ttotal_time_LIGHTTRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE))
#         print("\ttotal_num_FRAMES: {:d}".format(total_num_FRAMES))
#         print("\ttotal_num_PERSONS: {:d}\n".format(total_num_PERSONS))
#         print("\tAverage FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
#         print("\tAverage FPS excluding Pose Estimation: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_POSE)))
#         print("\tAverage FPS excluding Detection: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET)))
#         print("\tAverage FPS for framework only: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE)))

        # print the progress bar in stderr
        while (i + 1) / len(source_vframe_dir) >= progress_bar / 100:
            print(str(progress_bar) + '%', end='  ', flush=True, file=sys.stderr)
            if progress_bar >= 100:
                break
            progress_bar += 10
    print('', file=sys.stderr)
    
    print("###### List of videos with low number of predicted poses shown below. #######\n{}".format(low_num_skeletons)) ####