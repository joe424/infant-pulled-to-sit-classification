import os
import os.path as osp
import cv2
import numpy as np

import glob
import fnmatch

import json
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,5),(4,6),(5,6),(5,7),(6,8),(7,9),(8,10),(5,11),(6,12),
         (11,12),(11,13),(12,14),(13,15),(14,16)]
colors=['#FF3333','#FF5533','#FF7733','#FF9933','#FFBB33','#FFDD33',
        '#FFFF33','#DDFF33','#BBFF33','#99FF33','#77FF33','#55FF33',
        '#33FF33','#33FF55','#33FF77','#33FF99','#33FFBB','#33FFDD',
        '#33FFFF']

actions = ['supine', 'prone', 'pull']  
def draw_skeleton(joints, json_path):
    figure, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(left=0,right=1920,auto=False)
    ax.set_ylim(bottom=1080,top=0,auto=False)
    ax.text(10,30, json_path, fontsize=14)
    for i, edge in enumerate(edges):
        p1, p2 = edge
        plt.plot([joints[p1][0], joints[p2][0]], [joints[p1][1], joints[p2][1]], c=colors[i])
    
    for vertex in range(17):
        plt.scatter([joints[vertex][0]],[joints[vertex][1]],c=colors[vertex])

def extract_infant_skeleton(json_path, pkl_path):
    write_pkl_bool = True  # control if save extracted skeletons or not.
    index = 0               # indexing the ith predicted pose in a frame.
    manual_start = 0        # indexing the ith frame in a video.

    ## threshold 
    min_dist = 150  # minist distance between two center of bounding boxes in adjacent frames. 
    at_least_number_skeletons = 20
    # Optional parameters
    # skip videos matching kewords in {videos2skip_list}.
    video2skip_list = [] 
    ####

    json_name = os.path.basename(json_path)
    action = None
    for act in actions:
        if act in json_name.lower():
            action = act
            break
    output_filedir = pkl_path
#     output_filedir = '{}/{}/'.format(output_filedir, action)
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir, exist_ok=True)
    # else:
    #     print("File exist. {}".format(output_filedir))
        

    ## Preprocess of input/out file path
    for name in video2skip_list:
        if name in json_path:
            continue      
    output_baby_skeleton_filename = "{}/{}_infant.pkl".format(output_filedir,os.path.splitext(os.path.basename(json_path))[0])
    baby_skeleton_file = None
    if write_pkl_bool == True:
        baby_skeleton_file = open(output_baby_skeleton_filename, 'wb')
    
    
    infant_skeletons_history = OrderedDict()
    historylist_cnt = 0
    with open(json_path, newline='') as jsonfile:
        spamreader = json.load(jsonfile)
        first_skeleton = False
        for i, row in enumerate(spamreader):
            if len(row["candidates"]) <= 1 and row["candidates"][0]["det_score"] == 0 or i < manual_start:
                continue
            else:
                if first_skeleton == False:
                    first_skeleton = True
                    kps_2d = row["candidates"][index].get("pose_keypoints_2d")
                    head_infant_pose = np.array(kps_2d).reshape(17,3)[:,:2]
                    infant_skeletons_history[0] = head_infant_pose
                    # draw_skeleton(head_infant_pose, json_path)
                else:
                    for candidate in row["candidates"]:
                        candidate_pose = np.array(candidate["pose_keypoints_2d"]).reshape((17,3))[:,:2]
                        prev_infant_pose = infant_skeletons_history[historylist_cnt][:,:2]
                        
                        candidate_pose_center = np.mean(candidate_pose, axis=0)
                        prev_infant_center = np.mean(prev_infant_pose, axis=0)
                        
                        dist = np.sum((candidate_pose_center - prev_infant_center)**2)
                        dist = np.sqrt(dist)
                        if dist < min_dist:
                            historylist_cnt += 1
                            infant_skeletons_history[historylist_cnt] = candidate_pose
                            # if i - manual_start < 2:
                            #     draw_skeleton(candidate_pose, json_path)
                            break  
                  
    if historylist_cnt > at_least_number_skeletons:
        if write_pkl_bool == True:
            pickle.dump(infant_skeletons_history, baby_skeleton_file)
            baby_skeleton_file.close()
            print("\t[{}/{}, infant poses extracted] [Successed!]. Finished and stored in {}".format(historylist_cnt, len(spamreader),baby_skeleton_file))
        return True
    else:
        print("\t[{}/{} infant poses extracted] [Failed!]. Num_poses is less than {}".format(historylist_cnt, len(spamreader), at_least_number_skeletons))
        return False