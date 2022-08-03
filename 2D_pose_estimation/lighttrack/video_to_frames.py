# Convert video to frames
# present_path = !pwd

import os
import os.path as osp
import numpy as np
import sys
import cv2
from glob import glob
import argparse
####
# Enviroment
# Path: root/{infant_case}/{CA_mth}/{videos}
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', '-v', type=str, dest='video_dir', default="../videos")
parser.add_argument('--frame_dir', '-f', type=str, dest='frame_dir', default="../frames")
args = parser.parse_args()

# data_root = osp.abspath("/home/mbl/HDD_disk/infant_video/實驗收案")
# video_root = osp.join(data_root,'20210412_prone')
# destination_root = osp.join(data_root,"20210412_prone_frames")

video_root = args.video_dir
destination_root = args.frame_dir

# parameters 
frame_stride = 1  # Write a frame in every 5 frames. 
                  # e.q. if frame_stride = 5, a video(with 30 frames) --converted to--> 6 frames

# /home/mbl/HDD_disk/infant_video/實驗收案/HPE_dataset/out/hrnet/jsons
# check_root = "/home/mbl/HDD_disk/infant_video/實驗收案/HPE_dataset/"

# Optional parameters
# skip videos matching kewords in {videos2skip_list}.
videos2skip_list = [] 
####

video_paths = glob(osp.join(video_root, "*/*"))
total_num_vidoes = len(video_paths)
# print("########## Loading videos into images ##########") ####
# print("There are {} videos to process".format(total_num_vidoes)) ####
num_videos_converted = 0
progress_bar = 10
print('   video to frames: ', end='', flush=True, file=sys.stderr)
for idx, video_path in enumerate(video_paths):

    # Optional operation
    skip_flag = False 
    for v2skip_name in videos2skip_list:
        if v2skip_name in video_path:
            skip_flag = True
    if skip_flag:
        continue
    
    # Process destinate path  
    names = video_path.split('/')
    child_dir = osp.join(*names[-2:-1])
    
    video_name = osp.basename(video_path)
    pure_name = osp.splitext(video_name)[0] # name without pose-fix such as ".jpg" or ".mp4"
    
    new_frame_dir = osp.join(destination_root, child_dir, pure_name)
    # Check if the video has been processed before. 
#     check_dir = osp.join(check_root, destination_root, child_dir, pure_name)
    if os.path.exists(new_frame_dir): #or os.path.exists(check_dir) :
#         print("[Video frames exists] in ", new_frame_dir) ####
        continue
    else:
        os.makedirs(new_frame_dir)
        
    # Convert a video to frames
#     print("Reading video: {}\n Select 1 frame in every {} frames.".format(video_path, frame_stride)) ####
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    frame_seq = 0
    while success:
        if frame_seq % frame_stride == 0:
            cv2.imwrite("{}/{}_{:04d}.jpg".format(new_frame_dir, pure_name, frame_seq), image)
        success, image = cap.read()
        frame_seq += 1
    
    # Message
    num_videos_converted += 1
#     print("({:05d}/{:05d}) {} successfully converted.".format(num_videos_converted, total_num_vidoes, video_path)) ####
#     print("             Frames are stored in {}".format(new_frame_dir))
    
    # print the progress bar in stderr
    while (idx + 1) / len(video_paths) >= progress_bar / 100:
        print(str(progress_bar) + '%', end='  ', flush=True, file=sys.stderr)
        if progress_bar >= 100:
            break
        progress_bar += 10
print('', file=sys.stderr)