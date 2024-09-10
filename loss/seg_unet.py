import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from os.path import exists


class SegUNet_F(nn.Module):
    """
    loss_mode:
        label -> dice loss of segmentation results
    """

    def __init__(self, loss_layers, mode='OASIS'):
        super(SegUNet_F, self).__init__()

        unet_path = None
        in_channels = None
        classes = None
        if 'OASIS' in mode:
            unet_path = 'loss/unet_oasis.pt'
            in_channels = 1
            classes = 4
        elif 'BraTS' in mode:
            unet_path = 'loss/unet_brats.pt'
            in_channels = 4
            classes = 4
        elif 'ACDC' in mode:
            unet_path = 'loss/unet_acdc.pt'
            in_channels = 1
            classes = 4
        elif 'COVID' in mode:
            unet_path = 'loss/unet_covid.pt'
            in_channels = 1
            classes = 4

        dice_classes = [0, 1, 2, 3]
        if 'tumor_only' in mode:
            dice_classes = [1, 2, 3]
        if 'lesion_only' in mode:
            dice_classes = [1, 2, 3]

        unet = smp.Unet(in_channels=in_channels, classes=classes)
        if not exists(unet_path):
            raise ValueError('Pre-trained UNet not exist: {}'.format(unet_path))
        unet.load_state_dict(torch.load(unet_path, map_location='cpu'))

        for k in loss_layers:
            self.loss_mode = k
        self.loss_layers = loss_layers[self.loss_mode]

        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.tail = unet.segmentation_head
        self.softmax = nn.Softmax(1)

        self.encoder.requires_grad = False
        self.decoder.requires_grad = False
        self.tail.requires_grad = False

        self.loss_names = ['SegUNet({})'.format(self.loss_mode)]

        # reflection padding
        # padding (96, 96) to (160, 128), padding -> (32, 32, 16, 16)
        self.padding = nn.ReflectionPad2d((16, 16, 32, 32))
        self.padding_flag = False

        # dice loss if necessary
        if 'label' in self.loss_mode:
            self.loss = smp.losses.DiceLoss('multiclass', dice_classes)
        elif 'L1' in self.loss_mode:
            self.loss = torch.nn.MSELoss()
        elif 'L2' in self.loss_mode:
            self.loss = torch.nn.L1Loss()
        else:
            self.loss = torch.nn.L1Loss()

    def unet_forward(self, x):
        # padding sr/hr to [160, 128], as the trained UNet
        if self.padding_flag:
            x = self.padding(x)
        # 获取输入张量的高度和宽度
        height, width = x.shape[-2], x.shape[-1]
        # 计算下一个 2 的幂
        def next_power_of_2(n):
            return 2 ** (n - 1).bit_length()

        target_height = next_power_of_2(height)
        target_width = next_power_of_2(width)

        # 计算需要 padding 的数量
        pad_height = target_height - height
        pad_width = target_width - width

        # Padding 参数需要按照 (左, 右, 上, 下) 的顺序
        padding = (0, pad_width, 0, pad_height)

        # 应用 padding
        x = F.pad(x, padding, mode='constant', value=0)

        features = self.encoder(x)
        if 'encoder' in self.loss_mode:
          return features
        decoder_output, decoder_feature = self.decoder(*features)
        if 'decoder-feature' == self.loss_mode:
            return decoder_feature
        if 'decoder' in self.loss_mode:
          return decoder_output
        label = self.tail(decoder_output)
        softmax = self.softmax(label)
        if self.loss_mode in ['label-hr', 'label-gt', 'probability-map']:
            return label

    def forward(self, sr, hr, gt_label=None):

        if self.loss_mode != 'probability-map' and self.loss_mode != 'decoder-feature':
            assert sr.shape == hr.shape, 'Seg UNet Loss invalid SR({}) and HR({}) shape!'.format(
                sr.shape, hr.shape
            )
        else :
            assert hr == None and sr != None

        sr_features = self.unet_forward(sr)
        ret_feature = None

        if self.loss_mode == 'label-hr':
            with torch.no_grad():
                hr_label = self.unet_forward(hr)
            hr_label = torch.argmax(hr_label, dim=1)
            loss = self.loss(sr_features, hr_label)
        elif self.loss_mode == 'label-gt':
            if gt_label.dim() == 4:
                gt_label = gt_label[:, 0]
            gt_label = gt_label.to(torch.long)
            if self.padding_flag:
                gt_label = self.padding(gt_label)
            loss = self.loss(sr_features, gt_label)
        elif self.loss_mode == 'probability-map':
            ret_feature = self.softmax(sr_features)
            loss = None
        elif self.loss_mode == 'decoder-feature':
            ret_feature = sr_features
            loss = None
        elif 'encoder' in self.loss_mode:
            with torch.no_grad():
                hr_features = self.unet_forward(hr)
            loss = 0
            for l in self.loss_layers:
                loss += self.loss(sr_features[l], hr_features[l])
                loss /= len(self.loss_layers)
        elif 'decoder' in self.loss_mode:
            with torch.no_grad():
                hr_features = self.unet_forward(hr)
            loss = self.loss(sr_features, hr_features)
        else:
            raise ValueError('Invalid UNet Seg Loss Mode: {}'.format(self.loss_mode))

        if loss == None:
            loss_ret_val = 0
        else :
            loss_ret_val = loss.item()
        return loss, {self.loss_names[0]: loss_ret_val}, ret_feature
