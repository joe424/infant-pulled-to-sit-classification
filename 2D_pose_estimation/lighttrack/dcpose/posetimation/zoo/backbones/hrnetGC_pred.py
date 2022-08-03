#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
import torch.nn as nn
from posetimation.layers import Interpolate, BasicBlock, Bottleneck, ChainOfBasicBlocks, DeformableCONV, PAM_Module, CAM_Module
from thirdparty.deform_conv import DeformConv, ModulatedDeformConv
import logging
import os.path as osp

from ..base import BaseModel
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY

BN_MOMENTUM = 0.1
__all__ = ["HRNetGC_pred", "HighResolutionModule"]

class VIS(nn.Module):
    def __init__(self, in_channels, out_vector=4):
        super(VIS, self).__init__()
        # self.mid_channels = mid_channels
        self.bottleneck = BasicBlock(in_channels, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((17,17))
        self.conv1 = nn.Conv2d(in_channels, 1, 3, 1, 0)
        self.avg_visibility_predictor = nn.Linear(15*15,out_vector, bias=False)
        self.sigmoid =nn.Sigmoid()
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.avg_visibility_predictor(x.view(-1, 15*15))
        x = self.sigmoid(x)
        return x

class GCB(nn.Module):
    def __init__(self, num_channels, hw=None,reduce=4):
        super(GCB, self).__init__()
        self.H, self.W = hw
        self.relative_depth_condition_dim = 4
        self.new_num_channel = num_channels + self.relative_depth_condition_dim
        new_num_channel = self.new_num_channel
        self.context_weight = nn.Sequential(
            nn.Conv2d(new_num_channel, 1, 1, 1, 0),
            nn.Flatten(),
            nn.Softmax(),
        )
        # weight vector
        self.wv12 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduce, 1, 1, 0),
            nn.LayerNorm((num_channels // reduce, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduce, num_channels, 1, 1, 0),
        )

    def forward(self, x, rdc):  # N=2
        b, num_channels, h, w = x.size()
        condition = rdc.view(-1, 4, 1, 1).expand(-1,-1,h, w).clone()

        x_condition = torch.cat((x, condition), dim=1)
        y_weight = self.context_weight(x_condition).view(b, 1, h*w, 1)
        y_feature = x.view(b, num_channels, 1, h*w)
        y = torch.matmul(y_feature, y_weight)
        y = self.wv12(y)
        x = x + y
        return x

@MODEL_REGISTRY.register()
class HRNetGC_pred(BaseModel):
    def init_weights(self, *args, **kwargs):
        logger = logging.getLogger(__name__)

        # TODO Parameters are initialized through the distribution

        if osp.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                need_init_state_dict[name] = m

            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            # raise ValueError('{} is not exist!'.format(self.pretrained))

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        rot_factor = cfg.TRAIN.ROT_FACTOR
        SCALE_FACTOR = cfg.TRAIN.SCALE_FACTOR

        if not isinstance(SCALE_FACTOR, list):
            temp = SCALE_FACTOR
            SCALE_FACTOR = [SCALE_FACTOR, SCALE_FACTOR]
        scale_bottom = 1 - SCALE_FACTOR[0]
        scale_top = 1 + SCALE_FACTOR[1]

        paramer = "lr_{}_bbox_{}_rot_{}_scale_{}-{}".format("-5e5",bbox_enlarge_factor, rot_factor, scale_bottom, scale_top)

        return paramer

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = HRNetGC_pred(cfg, phase, **kwargs)
        return model

    def __init__(self, cfg, phase, **kwargs):
        super(HRNetGC_pred, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.is_train = True if phase == TRAIN_PHASE else False
        self.pretrained = cfg.MODEL.PRETRAINED

        self.inplanes = 64
        self.use_deconv = kwargs.get("use_deconv", False)

        self.freeze_hrnet_weight = cfg['MODEL']["FREEZE_HRNET_WEIGHTS"]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)

        branch1_hw = torch.tensor(cfg['MODEL']["HEATMAP_SIZE"])
        self.branch1_hw = torch.tensor([int(branch1_hw[0]), int(branch1_hw[1])])
        self.pred_head = VIS(pre_stage_channels[0])
        self.GCB_transition3_1 = GCB(num_channels=num_channels[0], hw=self.branch1_hw)
        self.GCB_transition3_2 = GCB(num_channels=num_channels[1], hw=self.branch1_hw/2)
        self.GCB_transition3_3 = GCB(num_channels=num_channels[2], hw=self.branch1_hw/4)
        self.GCB_transition3_4 = GCB(num_channels=num_channels[3], hw=self.branch1_hw/8)
        self.GCB_transition4 = GCB(num_channels=num_channels[0], hw=self.branch1_hw)

        self.stage4, self.pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
        
        self.final_layer = nn.Conv2d(
            in_channels=self.pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def forward(self, x, **kwargs):
        # rough heatmaps in sequence frames
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x1_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x1_list.append(self.transition1[i](x))
            else:
                x1_list.append(x)
        y_list = self.stage2(x1_list)
        
        x2_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x2_list.append(self.transition2[i](y_list[-1]))
            else:
                x2_list.append(y_list[i])
        y_list = self.stage3(x2_list)

        
        x3_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x3_list.append(self.transition3[i](y_list[-1]))
            else:
                x3_list.append(y_list[i])

        pred_head = self.pred_head(x3_list[0]) 
        gt_label = kwargs.get("depth_label", None)
        if gt_label != None:
            pred_head = kwargs["depth_label"]
        x3_list_new = [self.GCB_transition3_1(x3_list[0], pred_head)]
        x3_list_new.append(self.GCB_transition3_2(x3_list[1], pred_head))
        x3_list_new.append(self.GCB_transition3_3(x3_list[2], pred_head))
        x3_list_new.append(self.GCB_transition3_4(x3_list[3], pred_head))

        y_list = self.stage4(x3_list_new)
        y_list = self.GCB_transition4(y_list[0], pred_head)
        # y_list = self.deform(y_list)
        
        rough_pose_heatmaps = self.final_layer(y_list)


        if self.use_deconv:
            return y_list[0], rough_pose_heatmaps
        else:
            return rough_pose_heatmaps, pred_head

    def freeze_weight(self):
        for module in self.modules():
            parameters = module.parameters()
            for parameter in parameters:
                parameter.requires_grad = False

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _apply_channel_softmax(self):
        return nn.Softmax(dim=1)

    def _compute_chain_of_basic_blocks_3d(self, input_channel, out_channel, kh, kw, dd, dg, num_blocks):
        ## define stage
        num_blocks = num_blocks
        block = BasicBlock
        head_conv_input_channel = input_channel
        body_conv_input_channel = out_channel
        body_conv_output_channel = out_channel
        stride = 1

        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                head_conv_input_channel,
                body_conv_input_channel,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(
                body_conv_input_channel,
                momentum=BN_MOMENTUM
            ),
        )

        ##########3
        layers = []
        layers.append(
            block(
                head_conv_input_channel,
                body_conv_input_channel,
                stride,
                downsample
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                block(
                    body_conv_input_channel,
                    body_conv_output_channel
                )
            )
        return nn.Sequential(*layers)

    def _compute_chain_of_basic_blocks(self, nc, ic, kh, kw, dd, dg, b):

        if self.warper_mode == "2d":
            num_blocks = b
            block = BasicBlock
            in_ch = ic
            out_ch = ic
            stride = 1

            ######
            downsample = nn.Sequential(
                nn.Conv2d(
                    nc,
                    in_ch,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    in_ch,
                    momentum=BN_MOMENTUM
                ),
            )

            ##########3
            layers = []
            layers.append(
                block(
                    nc,
                    out_ch,
                    stride,
                    downsample
                )
            )

            for i in range(1, num_blocks):
                layers.append(
                    block(
                        in_ch,
                        out_ch
                    )
                )

            return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        logger = logging.getLogger(__name__)
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),

                            Interpolate(scale_factor=2 ** (j - i), mode='nearest')
                            # nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
