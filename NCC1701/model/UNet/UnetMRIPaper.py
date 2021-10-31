#!/usr/bin/env python

"""
Architecture of Unet 2D
Uses 2D Images
Based on: https://github.com/timctho/unet-pytorch/blob/master/Unet.py
With static hyper parameters
To call this UNet: net = UNet(n_channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Not tested"

class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, pool_size=2, do_batchnorm=True):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, kernel_size,  stride=stride, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size, stride=stride, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(pool_size, return_indices=True)
        self.relu = torch.nn.ReLU()
        self.do_batchnorm = do_batchnorm

    def forward(self, x):
        if self.do_batchnorm:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
        x_pooled, indices = self.max_pool(x)
        return x_pooled, x, indices

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, scale_factor=2, upsample_mode='nearest', kernel_size=3, stride=1, padding=1, do_batchnorm=True):
        super(UNet_up_block, self).__init__()
        self.max_unpool = torch.nn.MaxUnpool2d(2)
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, input_channel, kernel_size, stride=stride, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(input_channel)
        self.conv2 = torch.nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.do_batchnorm = do_batchnorm

    def forward(self, prev_feature_map, x, poolindices):
        x = self.max_unpool(x, poolindices)
        x = torch.cat((x, prev_feature_map), dim=1)
        if self.do_batchnorm:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
        return x


class UNet(torch.nn.Module):
    def __init__(self, n_channels, do_batchnorm=False):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(n_channels, 64)
        self.down_block2 = UNet_down_block(64, 128)

        self.mid_conv1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.mid_conv2 = torch.nn.Conv2d(256, 128, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)

        self.up_block1 = UNet_up_block(128, 128, 64)
        self.up_block2 = UNet_up_block(64, 64, 64)

        self.last_conv = torch.nn.Conv2d(64, n_channels, 1, padding=0)
        self.relu = torch.nn.ReLU()

        self.do_batchnorm = do_batchnorm

    def forward(self, x):
        x, x_notpooled1, poolindices1 = self.down_block1(x)
        x, x_notpooled2, poolindices2 = self.down_block2(x)
        if self.do_batchnorm:
            x = self.relu(self.bn1(self.mid_conv1(x)))
            x = self.relu(self.bn2(self.mid_conv2(x)))
        else:
            x = self.relu(self.mid_conv1(x))
            x = self.relu(self.mid_conv2(x))
        x = self.up_block1(x_notpooled2, x, poolindices2)
        x = self.up_block2(x_notpooled1, x, poolindices1)
        x = self.last_conv(x)
        return x