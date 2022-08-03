#!/usr/bin/python
# -*- coding:utf8 -*-

# model
from .build import build_model, get_model_hyperparameter

# # DcPose
# from .DcPose.dcpose_rsn import DcPose_RSN
# from .DcPose.dcpose_rsn_midLker_gintuition import DcPose_RSN_midLker_gintuition
# from .DcPose.dcpose_rsn_midLker_gstatistic import DcPose_RSN_midLker_gstatistic
# from .DcPose.dcpose_rsn_midLker_gstatistic_convmidfeat import DcPose_RSN_midLker_gstatistic_convmidfeat
# from .DcPose.dcpose_rsn_bbox_matching import DcPose_RSN_bbox_matching
# from .DcPose.dcpose_rsn_pred import DcPose_RSN_pred_propagate



# HRNet
from .backbones.hrnet import HRNet
# SimpleBaseline
from .backbones.simplebaseline import SimpleBaseline
