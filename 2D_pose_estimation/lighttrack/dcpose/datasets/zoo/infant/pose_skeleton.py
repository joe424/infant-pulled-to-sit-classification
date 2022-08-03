#!/usr/bin/python
# -*- coding:utf8 -*-

__all__ = ["PoseTrack_Official_Keypoint_Ordering",
           "PoseTrack_Keypoint_Pairs",
           "PoseTrack_Keypoint_Name_Colors",
           "PoseTrack_COCO_Keypoint_Ordering",
           "PoseTrack_COCO_Keypoint_Pair"]

#  PoseTrack Official Keypoint Ordering  - A total of 15
PoseTrack_Official_Keypoint_Ordering = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'head_bottom',
    'nose',
    'head_top',
]

# Endpoint1 , Endpoint2 , line_color
PoseTrack_Keypoint_Pairs = [
    ['head_top', 'head_bottom', 'Rosy'],
    ['head_bottom', 'right_shoulder', 'Yellow'],
    ['head_bottom', 'left_shoulder', 'Yellow'],
    ['right_shoulder', 'right_elbow', 'Blue'],
    ['right_elbow', 'right_wrist', 'Blue'],
    ['left_shoulder', 'left_elbow', 'Green'],
    ['left_elbow', 'left_wrist', 'Green'],
    ['right_shoulder', 'right_hip', 'Purple'],
    ['left_shoulder', 'left_hip', 'SkyBlue'],
    ['right_hip', 'right_knee', 'Purple'],
    ['right_knee', 'right_ankle', 'Purple'],
    ['left_hip', 'left_knee', 'SkyBlue'],
    ['left_knee', 'left_ankle', 'SkyBlue'],
]

# Facebook PoseTrack Keypoint Ordering (convert to COCO format)  -   A total of 17
PoseTrack_COCO_Keypoint_Ordering = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

PoseTrack_COCO_Keypoint_Pair = [
    ['nose', 'head_bottom', 'Rosy'],
    ['nose', 'head_top', 'Rosy'],
    ['head_bottom', 'left_ear', 'Yellow'],
    ['head_top', 'right_ear', 'Orange'],
    ['left_ear', 'left_shoulder', 'Yellow'],
    ['right_ear', 'right_shoulder', 'Orange'],
    ['left_shoulder', 'left_elbow', 'Blue'],
    ['right_shoulder', 'right_elbow', 'Green'],
    ['left_shoulder', 'left_hip','Purple'],
    ['right_shoulder', 'right_hip', 'SkyBlue'],
    ['left_elbow', 'left_wrist', 'Blue'],
    ['right_elbow', 'right_wrist', 'Green'],
    ['left_hip', 'left_knee','Purple'],
    ['right_hip', 'right_knee', 'SkyBlue'],
    ['left_knee', 'left_ankle','Purple'],
    ['right_knee', 'right_ankle', 'SkyBlue']]

PoseTrack_Keypoint_Name_Colors = [['right_ankle', 'Gold'],
                                  ['right_knee', 'Orange'],
                                  ['right_hip', 'DarkOrange'],
                                  ['left_hip', 'Peru'],
                                  ['left_knee', 'LightSalmon'],
                                  ['left_ankle', 'OrangeRed'],
                                  ['right_wrist', 'LightGreen'],
                                  ['right_elbow', 'LimeGreen'],
                                  ['right_shoulder', 'ForestGreen'],
                                  ['left_shoulder', 'DarkTurquoise'],
                                  ['left_elbow', 'Cyan'],
                                  ['left_wrist', 'PaleTurquoise'],
                                  ['head_bottom', 'DoderBlue'],
                                  ['nose', 'HotPink'],
                                  ['head_top', 'SlateBlue'],
                                  ['left_ear', 'Yellow'],
                                  ['right_ear', 'Orange'],]
