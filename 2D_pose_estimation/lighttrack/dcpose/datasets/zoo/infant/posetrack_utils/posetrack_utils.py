#!/usr/bin/python
# -*- coding:utf8 -*-

import os, json
import scipy.io as sio


def video2filenames(annot_dir, **kargs):
    pathtodir = annot_dir
    output, L = {}, {}
    if os.path.isfile(pathtodir) == False:
        total = annot_dir + '/val_0412PTS_5sec.json'
        with open(os.path.join(pathtodir, total), 'r') as fin:
            data = json.load(fin)
        fm_bg2end_dict = kargs.get("fm_bg2end_dict")
        dataset_ids = [dset_id for dset_id in fm_bg2end_dict]


        for fname in dataset_ids:
            if ext_types == '.json':
                video_name = data['images']
                temp = data['images'][0]['file_name']
                num_frames = data['images'][0]['nframes']

            else:
                raise NotImplementedError()
            video = os.path.dirname(temp)
            output[video] = out_fname
            L[video] = num_frames
        return output, L


    else:
        mat_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.mat' in f]
        json_files = [f for f in os.listdir(pathtodir) if
                    os.path.isfile(os.path.join(pathtodir, f)) and '.json' in f]

        if len(json_files) > 1:
            files = json_files
            ext_types = '.json'
        else:
            files = mat_files
            ext_types = '.mat'

    
        for fname in json_files:
            if ext_types == '.mat':
                out_fname = fname.replace('.mat', '.json')
                data = sio.loadmat(
                    os.path.join(pathtodir, fname), squeeze_me=True,
                    struct_as_record=False)
                temp = data['annolist'][0].image.name

                data2 = sio.loadmat(os.path.join(pathtodir, fname))
                num_frames = len(data2['annolist'][0])
            elif ext_types == '.json':
                out_fname = fname
                with open(os.path.join(pathtodir, fname), 'r') as fin:
                    data = json.load(fin)

                if 'annolist' in data:
                    temp = data['annolist'][0]['image'][0]['name']
                    num_frames = len(data['annolist'])
                else:
                    temp = data['images'][0]['file_name']
                    num_frames = data['images'][0]['nframes']


            else:
                raise NotImplementedError()
            video = os.path.dirname(temp)
            output[video] = out_fname
            L[video] = num_frames
        return output, L


