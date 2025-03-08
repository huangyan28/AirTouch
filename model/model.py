import math
import time
from termcolor import cprint

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings, BertEncoder, BertPooler,
                                                    BertPreTrainedModel)

from convNeXT.resnetUnet import convNeXTUnet, convNeXTUnet_RGB2offset_3D
from model.resnet import BasicBlock, Bottleneck
from model.resnetUnet import (OfficialResNetUnet, OfficialResNetUnet_RGB2offset_3D,
                              OfficialResNetUnet_RGB2offset_3D_FC)

BN_MOMENTUM = 0.1

resnet = {18: (BasicBlock, [2, 2, 2, 2]),
          50: (Bottleneck, [3, 4, 6, 3]),
          101: (Bottleneck, [3, 4, 23, 3]),
          152: (Bottleneck, [3, 8, 36, 3])
          }

class RGBDmodel(nn.Module):
    def __init__(self, net, pretrain, joint_num, dataset, mano_dir, kernel_size=1, mode='RGB'):
        super(RGBDmodel, self).__init__()  # Fix class name to RGBDmodel
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.classify_out = 3
        self.num_stages = 2
        self.net = net
        self.mode = mode  # Store mode to determine whether to use RGBD or just RGB

        # Define the backbone models for RGB and depth
        if 'convnext' in self.net:
            self.backbone_rgb = convNeXTUnet_RGB2offset_3D(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                                           out_dim_list=[joint_num * 3, joint_num, joint_num])
            if 'D' in self.mode:
                self.backbone_d = convNeXTUnet(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                               out_dim_list=[joint_num * 3, joint_num, joint_num])
        else:
            self.backbone_rgb = OfficialResNetUnet_RGB2offset_3D(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                                 out_dim_list=[joint_num * 3, joint_num, joint_num])
            if 'D' in self.mode:
                self.backbone_d = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                     out_dim_list=[joint_num * 3, joint_num, joint_num])

        # Define MLP layers for fusion (used when mode is RGBD)
        if self.mode == 'RGBD':
            self.fc_rgbd = nn.Sequential(
                nn.Linear(self.dim * 2, 512),   # input: concat RGB and depth features
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 63),  # 21 joints, each with 3 coordinates
            )

        cprint(f'Loading RGBDmodel in {self.mode} mode...', 'green')

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, img_rgb, img=None, pcl = None, kernel=0.8, writer=None, ii=0):

        if 'RGB' in self.mode:
            # 获取 RGB 图像的特征
            img_offset_rgb, img_feat_rgb = self.backbone_rgb(img_rgb)  # 这里处理 RGB 图像

        # 如果 mode 是 D 或 RGBD（有 D backbone）
        if 'D' in self.mode and img is not None:
            # 获取深度图像的特征
            img_offset_d, img_feat_d = self.backbone_d(img)  # 这里处理深度图像

        # 如果是 'RGBD' 模式，进行 RGB 和深度图像的融合
        if 'RGBD' in self.mode:
            # 检查尺寸是否一致，确保深度和 RGB 特征的尺寸相同
            if img_feat_rgb.size(2) != img_feat_d.size(2) or img_feat_rgb.size(3) != img_feat_d.size(3):
                raise ValueError(f"Feature map dimensions do not match! RGB size: {img_feat_rgb.size()}, Depth size: {img_feat_d.size()}")

            if img_feat_rgb.size(1) != img_feat_d.size(1):
                raise ValueError(f"Feature channels do not match! RGB channels: {img_feat_rgb.size(1)}, Depth channels: {img_feat_d.size(1)}")

            # 融合 RGB 和深度特征
            fused_features = torch.cat((img_feat_rgb, img_feat_d), dim=1)  # Shape: [B, C_rgb + C_d, H, W]

            # 可选的：使用全局平均池化处理融合特征
            pooled_features = F.adaptive_avg_pool2d(fused_features, (1, 1))  # Shape: [B, C_fused, 1, 1]
            pooled_features = pooled_features.view(img_feat_rgb.size(0), -1)  # Flatten to [B, C_fused]

            # 最终通过全连接层获取输出 (B, 63) -> (21, 3)
            final_output = self.fc_rgbd(pooled_features)

            # 重塑输出为 (B, 21, 3) 来表示 21 个关节和它们的 3D 坐标
            final_output = final_output.view(img_feat_rgb.size(0), 21, 3)

        elif 'RGB' in self.mode:
            # 仅处理 RGB 图像
            final_output = img_offset_rgb.view(img_feat_rgb.size(0), 21, 3)

        elif 'D' in self.mode:
            # 仅处理深度图像
            final_output = img_offset_d.view(img_feat_d.size(0), 21, 3)

        # 返回最终结果
        result = [final_output]
        return result, None, None
    
class Glovemodel(nn.Module):
    def __init__(self, net, pretrain, joint_num, dataset, mano_dir, kernel_size=1, mode='RGB'):
        super(Glovemodel, self).__init__()  # Fix class name to RGBDmodel
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.classify_out = 3
        self.num_stages = 2
        self.net = net
        self.mode = mode  # Store mode to determine whether to use RGBD or just RGB

        # Define the backbone models for RGB and depth
        if 'convnext' in self.net:
            self.backbone_rgb = convNeXTUnet_RGB2offset_3D(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                                           out_dim_list=[joint_num * 3, joint_num, joint_num])
            if 'D' in self.mode:
                self.backbone_d = convNeXTUnet(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                               out_dim_list=[joint_num * 3, joint_num, joint_num])
        else:
            self.backbone_rgb = OfficialResNetUnet_RGB2offset_3D(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                                 out_dim_list=[joint_num * 3, joint_num, joint_num],joint_out_dim = 1)
            if 'D' in self.mode:
                self.backbone_d = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                     out_dim_list=[joint_num * 3, joint_num, joint_num])

        # Define MLP layers for fusion (used when mode is RGBD)
        if self.mode == 'RGBD':
            self.fc_rgbd = nn.Sequential(
                nn.Linear(self.dim * 2, 512),   # input: concat RGB and depth features
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 63),  # 21 joints, each with 3 coordinates
            )

        cprint(f'Loading RGBDmodel in {self.mode} mode...', 'green')

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, img_rgb, img=None, pcl = None, kernel=0.8, writer=None, ii=0):

        if 'RGB' in self.mode:
            # 获取 RGB 图像的特征
            img_offset_rgb, img_feat_rgb = self.backbone_rgb(img_rgb)  # 这里处理 RGB 图像

        # 如果 mode 是 D 或 RGBD（有 D backbone）
        if 'D' in self.mode and img is not None:
            # 获取深度图像的特征
            img_offset_d, img_feat_d = self.backbone_d(img)  # 这里处理深度图像

        # 如果是 'RGBD' 模式，进行 RGB 和深度图像的融合
        if 'RGBD' in self.mode:
            # 检查尺寸是否一致，确保深度和 RGB 特征的尺寸相同
            if img_feat_rgb.size(2) != img_feat_d.size(2) or img_feat_rgb.size(3) != img_feat_d.size(3):
                raise ValueError(f"Feature map dimensions do not match! RGB size: {img_feat_rgb.size()}, Depth size: {img_feat_d.size()}")

            if img_feat_rgb.size(1) != img_feat_d.size(1):
                raise ValueError(f"Feature channels do not match! RGB channels: {img_feat_rgb.size(1)}, Depth channels: {img_feat_d.size(1)}")

            # 融合 RGB 和深度特征
            fused_features = torch.cat((img_feat_rgb, img_feat_d), dim=1)  # Shape: [B, C_rgb + C_d, H, W]

            # 可选的：使用全局平均池化处理融合特征
            pooled_features = F.adaptive_avg_pool2d(fused_features, (1, 1))  # Shape: [B, C_fused, 1, 1]
            pooled_features = pooled_features.view(img_feat_rgb.size(0), -1)  # Flatten to [B, C_fused]

            # 最终通过全连接层获取输出 (B, 63) -> (21, 3)
            final_output = self.fc_rgbd(pooled_features)

            # 重塑输出为 (B, 21, 3) 来表示 21 个关节和它们的 3D 坐标
            final_output = final_output.view(img_feat_rgb.size(0), 12, 1)

        elif 'RGB' in self.mode:
            # 仅处理 RGB 图像
            final_output = img_offset_rgb.view(img_feat_rgb.size(0), 12, 1)

        elif 'D' in self.mode:
            # 仅处理深度图像
            final_output = img_offset_d.view(img_feat_d.size(0), 12, 1)

        # 返回最终结果
        result = [final_output]
        return result, None, None