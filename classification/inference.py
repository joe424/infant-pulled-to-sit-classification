from glob import glob
import numpy as np
import os
import torch
import torch.nn as nn
import time
import random
from torch.utils.data import TensorDataset, DataLoader
import math
import torch.utils.data as data
import torch.optim as optim
import fnmatch
import pandas as pd
import pickle
from collections import OrderedDict
from tqdm import tqdm
from model.model import Model
import scipy.signal
import scipy.ndimage
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# parameters
is2d = True
num_class = 3

num_point = 3
angle_hint = True

# num_point = 5
# angle_hint = False

if num_point == 5:
    joints = ['nose', 'rshoulder', 'lshoulder', 'rhip', 'lhip']
    graph_file = "graph.five_joints.Graph"
elif num_point == 13:
    joints = ['rhip', 'rknee', 'rfoot', 'lhip', 'lknee', 'lfoot', 'nose', 'lshoulder', 'lelbow', 'lwrist', 'rshoulder', 'relbow', 'rwrist']
    graph_file = "graph.thirteen_joints.Graph"
elif num_point == 3: # unitization
    joints = ['nose', 'shoulder', 'hip']
    graph_file = "graph.three_joints.Graph"
else:
    assert 0, 'not support num_joint'

dims = ['x', 'y'] if is2d else ['x', 'y', 'z']
device = 1
torch.cuda.set_device(device)
in_channels = len(dims) + 1 if angle_hint else len(dims)
out_channels = 20
frame_num = 96
frames = [frame for frame in range(frame_num)]

if is2d and num_point == 5:
    joint_name_2_number = {'nose': 6, 'rshoulder': 10, 'lshoulder': 7, 'rhip': 0, 'lhip': 3}
elif is2d and num_point == 13:
    joint_name_2_number = {'rhip': 0, 'rknee': 1, 'rfoot': 2, 'lhip': 3, 'lknee': 4, 'lfoot': 5, 'nose': 6, 'lshoulder': 7, 'lelbow': 8, 'lwrist': 9, 'rshoulder': 10, 'relbow': 11, 'rwrist': 12}
elif not is2d and num_point == 5:
    joint_name_2_number = {'nose': 14, 'rshoulder': 25, 'lshoulder': 17, 'rhip': 1, 'lhip': 6}
elif not is2d and num_point == 13:
    joint_name_2_number = {'rhip': 1, 'rknee': 2, 'rfoot': 3, 'lhip': 6, 'lknee': 7, 'lfoot': 8, 'nose': 14, 'lshoulder': 17, 'lelbow': 18, 'lwrist': 19, 'rshoulder': 25, 'relbow': 26, 'rwrist': 27}
elif is2d and num_point == 3:
    joint_name_2_number = {'nose': 6, 'shoulder': [7, 10], 'hip': [0, 3]}
else:
    assert 0, 'not support is2d and num_point'

npy_path = './samples'
weight_path = './weight/fold2_2022.pt'

model = Model(num_class=num_class, num_point=num_point, num_person=1, graph=graph_file,
          in_channels=in_channels, out_channels=out_channels, frames=frame_num).cuda()
model.load_state_dict(torch.load(weight_path))
criterion = nn.BCEWithLogitsLoss().cuda()

# run model
with torch.no_grad():
    model.eval()
    for npy in sorted(glob(os.path.join(npy_path, '*'))):
        sk2d = np.load(npy)

        if angle_hint:
            data = np.empty([1, len(frames), len(joints), len(dims)+1], dtype='float64')
        else:
            data = np.empty([1, len(frames), len(joints), len(dims)], dtype='float64')

        for frame in frames:
            for joint_idx, joint in enumerate(joints):
                for dim_idx, dim in enumerate(dims):
                    if isinstance(joint_name_2_number[joint], list):
                        data[0][frame][joint_idx][dim_idx] = (sk2d[frame][joint_name_2_number[joint][0]][dim_idx] + sk2d[frame][joint_name_2_number[joint][1]][dim_idx]) / 2
                    else:
                        data[0][frame][joint_idx][dim_idx] = sk2d[frame][joint_name_2_number[joint]][dim_idx]
            if angle_hint:
                nose = [
                    data[0][frame][0][0],
                    data[0][frame][0][1]
                ]
                if num_point == 3: # unitization
                    shoulder = [
                        data[0][frame][1][0],
                        data[0][frame][1][1]
                    ]
                    hip = [
                        data[0][frame][2][0],
                        data[0][frame][2][1]
                    ]
                else:
                    shoulder = [
                        (data[0][frame][1][0] + data[0][frame][2][0]) / 2,
                        (data[0][frame][1][1] + data[0][frame][2][1]) / 2
                    ]
                    hip = [
                        (data[0][frame][3][0] + data[0][frame][4][0]) / 2,
                        (data[0][frame][3][1] + data[0][frame][4][1]) / 2
                    ]
                shoulder2nose = np.array(nose) - np.array(shoulder)
                hip2shoulder = np.array(shoulder) - np.array(hip)
                angle = np.arctan2(shoulder2nose[1], shoulder2nose[0]) - np.arctan2(hip2shoulder[1], hip2shoulder[0])
                data[0, frame, :, 2] = angle

                if num_point == 3: # unitization
                    data[0][frame][0][0] -= data[0][frame][1][0]
                    data[0][frame][0][1] -= data[0][frame][1][1]
                    data[0][frame][1][0] -= data[0][frame][1][0]
                    data[0][frame][1][1] -= data[0][frame][1][1]
                    data[0][frame][2][0] -= data[0][frame][1][0]
                    data[0][frame][2][1] -= data[0][frame][1][1]

                    data[0][frame][0][0] /= np.linalg.norm(shoulder2nose)
                    data[0][frame][0][1] /= np.linalg.norm(shoulder2nose)
                    data[0][frame][2][0] /= np.linalg.norm(hip2shoulder)
                    data[0][frame][2][1] /= np.linalg.norm(hip2shoulder)
        x = torch.from_numpy(data)
        x = x.unsqueeze(4).permute(0, 3, 1, 2, 4).float().cuda()
        y_pred = model(x)
        result = torch.argmax(y_pred.cpu(), dim=1).item()
        if result == 2:
            result += 1
        print(os.path.splitext(os.path.basename(npy))[0].split('_infant')[0] + ':\t\tlevel', result)
