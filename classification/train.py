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
from time import localtime, strftime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def summary(CM, num_class):
    # calculate accuracy
    sum_TP = 0
    n = np.sum(CM)
    for i in range(num_class):
        sum_TP += CM[i, i]
    acc = sum_TP / n

    # kappa
    sum_po = 0
    sum_pe = 0
    for i in range(len(CM[0])):
        sum_po += CM[i][i]
        row = np.sum(CM[i, :])
        col = np.sum(CM[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    kappa = round((po - pe) / (1 - pe), 3)

    Precision_list = []
    Recall_list = []
    Specificity_list = []
    for i in range(num_class):
        TP = CM[i, i]
        FP = np.sum(CM[i, :]) - TP
        FN = np.sum(CM[:, i]) - TP
        TN = np.sum(CM) - TP - FP - FN

        Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
        Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
        Precision_list.append(Precision)
        Recall_list.append(Recall)
        Specificity_list.append(Specificity)
    return acc, Precision_list, Recall_list, Specificity_list

total_time_start = time.time()

setup_seed(2022)

# parameters
weight_folder = './weight'
is2d = True
num_class = 3 # is target class or not
angle_hint = True
if angle_hint and not is2d:
    assert 0, 'not support yet'
target_level = '0' # not used if num_class = 3
SAME_FRAME = 96
levels = ['L0', 'L1', 'L3']
frames = [frame for frame in range(SAME_FRAME)]
num_point = 3
device = 0
golden_standard_equally_distributed = True

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

batch_size = 32
TURNS = 5 # 5 fold cross validation
turns = [turn for turn in range(TURNS)]
group_num = TURNS
level_count = {'L0': 0, 'L1': 0, 'L3': 0}
epochs = 100
lr = 0.05

torch.cuda.set_device(device)
in_channels = len(dims) + 1 if angle_hint else len(dims)
out_channels = 20
frame_num = SAME_FRAME
milestones = [30, 50, 70]

y_pred_turn = torch.empty((0)) ###
y_actual_turn = torch.empty((0)) ###

remove = ['19_3m_Pull_to_sit_1_41', '117_2m_Pull_to_sit_1_323', '658042906.989491_94'] # file that is ambiguous
filepath_list = []

if golden_standard_equally_distributed:
    golden_standard_dict = {'L0': [], 'L1': [], 'L3': []}
    golden_standard_dict_full_path = {'L0': [], 'L1': [], 'L3': []}


    # make golden standard to dict
    with open('./KMU_hospital_infant_pulled_to_sit_data/golden_standard_lateral_only.txt') as file:
        lines = [line.rstrip() for line in file]
    now_level = ''
    for i in lines:
        if i == 'L0' or i == 'L1' or i == 'L3':
            now_level = i
        else:
            if i in remove:
                continue
            golden_standard_dict[now_level].append(i)
    golden_standard_dict['L0'] = set(golden_standard_dict['L0'])
    golden_standard_dict['L1'] = set(golden_standard_dict['L1'])
    golden_standard_dict['L3'] = set(golden_standard_dict['L3'])

if is2d:
    filepath_list = sorted(glob(os.path.join('./KMU_hospital_infant_pulled_to_sit_data/pulled-to-sit/*/2dnpy_smoothed', '*.npy')))
else:
    filepath_list = sorted(glob(os.path.join('./KMU_hospital_infant_pulled_to_sit_data/pulled-to-sit/*/3dnpy_smoothed', '*.npy')))
    
# remove ambiguous file
tmp_rm = []
for filepath in filepath_list:
    for rm in remove:
        if filepath.find(rm) != -1:
            tmp_rm.append(filepath)
for rm in tmp_rm:
    filepath_list.remove(rm)

for filepath in filepath_list:
    if filepath.find('/0/') != -1:
        level_count['L0'] += 1
    elif filepath.find('/1/') != -1:
        level_count['L1'] += 1
    elif filepath.find('/3/') != -1:
        level_count['L3'] += 1
filepath_list_L0 = filepath_list[ : level_count['L0']]
filepath_list_L1 = filepath_list[level_count['L0'] : level_count['L0'] + level_count['L1']]
filepath_list_L3 = filepath_list[level_count['L0'] + level_count['L1'] : level_count['L0'] + level_count['L1'] + level_count['L3']]
random.shuffle(filepath_list_L0)
random.shuffle(filepath_list_L1)
random.shuffle(filepath_list_L3)
filepath_list = filepath_list_L0 + filepath_list_L1 + filepath_list_L3

# training and testing
#     model_dict = {}
best_test_accuracy = [0 for turn in range(TURNS)]
inference_time_turn_list = []

cur_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()) # for save weight name

acc_list, sen0_list, sen1_list, sen3_list, spec0_list, spec1_list, spec3_list = [], [], [], [], [], [], []
for turn in turns:
    
    model = Model(num_class=num_class, num_point=num_point, num_person=1, graph=graph_file,
              in_channels=in_channels, out_channels=out_channels, frames=frame_num).cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7, last_epoch=-1) ####
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7, last_epoch=-1) ####
    criterion = nn.BCEWithLogitsLoss().cuda()

    filepath_dict = {'L0': None, 'L1': None, 'L3': None}
    filepath_dict_train = {'L0': None, 'L1': None, 'L3': None}
    filepath_dict_test = {'L0': None, 'L1': None, 'L3': None}
    filepath_list_train = []
    filepath_list_test = []

    filepath_dict['L0'] = filepath_list[:level_count['L0']]
    filepath_dict['L1'] = filepath_list[level_count['L0'] : level_count['L0'] + level_count['L1']]
    filepath_dict['L3'] = filepath_list[level_count['L0'] + level_count['L1'] : level_count['L0'] + level_count['L1'] + level_count['L3']]

    # split data
    for level in levels:
        num_per_group = math.ceil(len(filepath_dict[level]) / group_num)
        filepath_dict_test[level] = filepath_dict[level][num_per_group * turn : num_per_group * (turn + 1)]
        filepath_dict_train[level] = filepath_dict[level].copy()
        for i in range(len(filepath_dict_test[level])):
            filepath_dict_train[level].remove(filepath_dict_test[level][i])
        for i in filepath_dict_train[level]:
            filepath_list_train.append(i)
        for i in filepath_dict_test[level]:
            filepath_list_test.append(i)

        if golden_standard_equally_distributed:
            num_per_group = math.ceil(len(golden_standard_dict_full_path[level]) / group_num)
            filepath_dict_test[level] = golden_standard_dict_full_path[level][num_per_group * turn : num_per_group * (turn + 1)]
            filepath_dict_train[level] = golden_standard_dict_full_path[level].copy()
            for i in range(len(filepath_dict_test[level])):
                filepath_dict_train[level].remove(filepath_dict_test[level][i])
            for i in filepath_dict_train[level]:
                filepath_list_train.append(i)
            for i in filepath_dict_test[level]:
                filepath_list_test.append(i)
                
    # prepare TRAINING data
    if angle_hint:
        if num_class == 3:
            x, y = np.empty([0, len(frames), len(joints), len(dims)+1], dtype='float64'), np.empty([0, 3], dtype='float64') ###
        elif num_class == 1:
            x, y = np.empty([0, len(frames), len(joints), len(dims)+1], dtype='float64'), np.empty([0, 1], dtype='float64') ###
        else:
            assert 0, 'num_class not support'
    else:
        if num_class == 3:
            x, y = np.empty([0, len(frames), len(joints), len(dims)], dtype='float64'), np.empty([0, 3], dtype='float64') ###
        elif num_class == 1:
            x, y = np.empty([0, len(frames), len(joints), len(dims)], dtype='float64'), np.empty([0, 1], dtype='float64') ###
        else:
            assert 0, 'num_class not support'

    for file in filepath_list_train:
        if is2d:
            level = file[file.find('2dnpy') - 2]
        else:
            level = file[file.find('3dnpy') - 2]

        sk2d = np.load(file)
        if is2d:
            sk2d = sk2d.reshape(SAME_FRAME, 13, len(dims))
        else:
            sk2d = sk2d.reshape(SAME_FRAME, 32, len(dims))

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

        x = np.concatenate((x, data))

        if num_class == 3:
            if level == '0':
                y = np.concatenate((y, [[1, 0, 0]])) ###
            elif level == '1':
                y = np.concatenate((y, [[0, 1, 0]])) ###
            elif level == '3':
                y = np.concatenate((y, [[0, 0, 1]])) ###
        elif num_class == 1:
            if level == target_level:
                y = np.concatenate((y, [[1]])) ###
            else:
                y = np.concatenate((y, [[0]])) ###
        else:
            assert 0, 'num_class not support'

    td = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_dataloader = DataLoader(td, batch_size=batch_size, shuffle=True) ###


    # prepare TESTING data
    if angle_hint:
        if num_class == 3:
            x, y = np.empty([0, len(frames), len(joints), len(dims)+1], dtype='float64'), np.empty([0, 3], dtype='float64') ###
        elif num_class == 1:
            x, y = np.empty([0, len(frames), len(joints), len(dims)+1], dtype='float64'), np.empty([0, 1], dtype='float64') ###
        else:
            assert 0, 'num_class not support'
    else:
        if num_class == 3:
            x, y = np.empty([0, len(frames), len(joints), len(dims)], dtype='float64'), np.empty([0, 3], dtype='float64') ###
        elif num_class == 1:
            x, y = np.empty([0, len(frames), len(joints), len(dims)], dtype='float64'), np.empty([0, 1], dtype='float64') ###
        else:
            assert 0, 'num_class not support'

    for file in filepath_list_test:
        if is2d:
            level = file[file.find('2dnpy') - 2]
        else:
            level = file[file.find('3dnpy') - 2]

        sk2d = np.load(file)
        if is2d:
            sk2d = sk2d.reshape(SAME_FRAME, 13, len(dims))
        else:
            sk2d = sk2d.reshape(SAME_FRAME, 32, len(dims))

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

        x = np.concatenate((x, data))

        if num_class == 3:
            if level == '0':
                y = np.concatenate((y, [[1, 0, 0]])) ###
            elif level == '1':
                y = np.concatenate((y, [[0, 1, 0]])) ###
            elif level == '3':
                y = np.concatenate((y, [[0, 0, 1]])) ###
        elif num_class == 1:
            if level == target_level:
                y = np.concatenate((y, [[1]])) ###
            else:
                y = np.concatenate((y, [[0]])) ###
        else:
            assert 0, 'num_class not support'

    td = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    test_dataloader = DataLoader(td, batch_size=2, shuffle=True) ###

    y_pred_epoch = torch.empty((0)) ###
    y_actual_epoch = torch.empty((0)) ###

    # run model
    inference_time_turn = 0
    best_train_accuracy = 0
    for epoch in range(epochs):
        y_pred_tmp = torch.empty((0)) ###
        y_actual_tmp = torch.empty((0)) ###
        train_accuracy, train_loss, count = 0, 0, 0
        count = 0
        model.train()
        for x, y_actual in train_dataloader:
            count += x.shape[0]
            x = torch.unsqueeze(x, 4).permute(0, 3, 1, 2, 4).float().cuda()
            y_actual = y_actual.float().cuda()

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_actual)
            loss.backward()
            optimizer.step()

            if num_class == 3:
                train_accuracy += (torch.argmax(y_pred, dim=1) == torch.argmax(y_actual, dim=1)).sum().item() ###
            elif num_class == 1:
                train_accuracy += ((torch.sigmoid(y_pred).data.round().long()) == y_actual).sum().item() ###
            else:
                assert 0, 'num_class not support'

            train_loss += loss.item()
        train_accuracy /= count

        # testing
        test_accuracy, test_loss, count = 0, 0, 0
        time_start = time.time()
        with torch.no_grad():
            model.eval()
            for x, y_actual in test_dataloader:
                count += x.shape[0]
                x = torch.unsqueeze(x, 4).permute(0, 3, 1, 2, 4).float().cuda()
                y_actual = y_actual.float().cuda()
                y_pred = model(x)
                loss = criterion(y_pred, y_actual)

                if num_class == 3:
                    y_pred_tmp = torch.cat((y_pred_tmp, torch.argmax(y_pred.cpu(), dim=1))) ###
                    y_actual_tmp = torch.cat((y_actual_tmp, torch.argmax(y_actual.cpu(), dim=1))) ###
                    test_accuracy += (torch.argmax(y_pred, dim=1) == torch.argmax(y_actual, dim=1)).sum().item() ###
                elif num_class == 1:
                    y_pred_tmp = torch.cat((y_pred_tmp, torch.sigmoid(y_pred.cpu()).data.round().long())) ###
                    y_actual_tmp = torch.cat((y_actual_tmp, torch.sigmoid(y_actual.cpu()).data.round().long())) ###
                    test_accuracy += ((torch.sigmoid(y_pred).data.round().long()) == y_actual).sum().item() ###
                else:
                    assert 0, 'num_class not support'

                test_loss += loss.item()
        inference_time_turn += (time.time() - time_start)
        test_accuracy /= count
        scheduler.step()

        # print result, pick the best test accuracy as result and save the model
        if test_accuracy > best_test_accuracy[turn]:
            y_pred_epoch = y_pred_tmp ###
            y_actual_epoch = y_actual_tmp ###
            best_test_accuracy[turn] = test_accuracy

            best_train_accuracy = train_accuracy

            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, weight_folder + '/' + cur_time + '_fold' + str(turn+1) + '.pt')

            print("Epoch {:>3d}, train loss: {:>5.3f}, train acc: {:>5.3f}%, test loss: {:>5.3f}, test acc: {:>5.3f}%, test best acc: {:>5.3f}%".format(epoch + 1, train_loss, train_accuracy * 100., test_loss, test_accuracy * 100., best_test_accuracy[turn] * 100.))
            
        elif test_accuracy == best_test_accuracy[turn] and best_train_accuracy < train_accuracy:
            y_pred_epoch = y_pred_tmp ###
            y_actual_epoch = y_actual_tmp ###

            best_train_accuracy = train_accuracy

            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, weight_folder + '/' + cur_time + '_fold' + str(turn+1) + '.pt')

            print("Epoch {:>3d}, train loss: {:>5.3f}, train acc: {:>7.3f}%, test loss: {:>5.3f}, test acc: {:>5.3f}%, test best acc: {:>5.3f}%".format(epoch + 1, train_loss, train_accuracy * 100., test_loss, test_accuracy * 100., best_test_accuracy[turn] * 100.))
            
        else:
            print("Epoch {:>3d}, train loss: {:>5.3f}, train acc: {:>5.3f}%, test loss: {:>5.3f}, test acc: {:>5.3f}%, test best acc: {:>5.3f}%".format(epoch + 1, train_loss, train_accuracy * 100., test_loss, test_accuracy * 100., best_test_accuracy[turn] * 100.))

    inference_time_turn_list.append(inference_time_turn / epochs / len(y_pred_epoch))
    y_pred_turn = torch.cat((y_pred_turn, y_pred_epoch)) ###
    y_actual_turn = torch.cat((y_actual_turn, y_actual_epoch)) ###
    print()
    print('='*50)
    print()
    
    acc, _, sensitivity_list, specificity_list = summary(confusion_matrix(y_pred_epoch, y_actual_epoch), num_class)
    acc_list.append(acc)
    sen0_list.append(sensitivity_list[0])
    sen1_list.append(sensitivity_list[1])
    sen3_list.append(sensitivity_list[2])
    spec0_list.append(specificity_list[0])
    spec1_list.append(specificity_list[1])
    spec3_list.append(specificity_list[2])

print('accuracy       ', sum(acc_list) / len(acc_list), '+-', max(sum(acc_list) / len(acc_list) - min(acc_list), max(acc_list) - sum(acc_list) / len(acc_list)))
print('sensitivity l0 ', sum(sen0_list) / len(sen0_list), '+-', max(sum(sen0_list) / len(sen0_list) - min(sen0_list), max(sen0_list) - sum(sen0_list) / len(sen0_list)))
print('sensitivity l1 ', sum(sen1_list) / len(sen1_list), '+-', max(sum(sen1_list) / len(sen1_list) - min(sen1_list), max(sen1_list) - sum(sen1_list) / len(sen1_list)))
print('sensitivity l3 ', sum(sen3_list) / len(sen3_list), '+-', max(sum(sen3_list) / len(sen3_list) - min(sen3_list), max(sen3_list) - sum(sen3_list) / len(sen3_list)))
print('specificity l0 ', sum(spec0_list) / len(spec0_list), '+-', max(sum(spec0_list) / len(spec0_list) - min(spec0_list), max(spec0_list) - sum(spec0_list) / len(spec0_list)))
print('specificity l1 ', sum(spec1_list) / len(spec1_list), '+-', max(sum(spec1_list) / len(spec1_list) - min(spec1_list), max(spec1_list) - sum(spec1_list) / len(spec1_list)))
print('specificity l3 ', sum(spec3_list) / len(spec3_list), '+-', max(sum(spec3_list) / len(spec3_list) - min(spec3_list), max(spec3_list) - sum(spec3_list) / len(spec3_list)))
print('inference time ', sum(inference_time_turn_list) / len(inference_time_turn_list))

CM = confusion_matrix(y_pred_turn, y_actual_turn, normalize='true') ###
if num_class == 3:
    display_labels = ['L0', 'L1', 'L3']
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=display_labels) ###
disp.plot(cmap=plt.cm.Blues,values_format='g') ###
plt.show() ###
    
print('total time is', time.time() - total_time_start, 's')
print()