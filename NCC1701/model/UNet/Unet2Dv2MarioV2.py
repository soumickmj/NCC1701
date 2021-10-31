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

class UNet(torch.nn.Module):
    def __init__(self, n_channels, do_batchnorm=False, kernel=9, stride=1, padding=4):
        super(UNet, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(n_channels, 32, kernel,  stride=stride, padding=padding)
        self.bn32 = torch.nn.BatchNorm2d(32)
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel,  stride=stride, padding=padding)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel,  stride=stride, padding=padding)
        self.bn64 = torch.nn.BatchNorm2d(64)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel,  stride=stride, padding=padding)
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel,  stride=stride, padding=padding)
        self.bn128 = torch.nn.BatchNorm2d(128)
        self.conv3_2 = torch.nn.Conv2d(128, 64, kernel,  stride=stride, padding=padding)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4_1 = torch.nn.Conv2d(128, 64, kernel,  stride=stride, padding=padding)
        self.conv4_2 = torch.nn.Conv2d(64, 32, kernel,  stride=stride, padding=padding)
        self.conv5_1 = torch.nn.Conv2d(64, 32, kernel,  stride=stride, padding=padding)
        self.conv5_2 = torch.nn.Conv2d(32, 32, kernel,  stride=stride, padding=padding)
        self.conv6 = torch.nn.Conv2d(32, n_channels, kernel,  stride=stride, padding=padding)

        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        

        self.do_batchnorm = do_batchnorm

    def forward(self, x):
        c1 = self.relu(self.bn32(self.conv1_1(x)))
        c1 = self.relu(self.bn32(self.conv1_2(c1)))
        p1 = self.relu(self.maxpool(c1))
        c2 = self.relu(self.bn64(self.conv2_1(p1)))
        c2 = self.relu(self.bn64(self.conv2_2(c2)))
        p2 = self.relu(self.maxpool(c2))
        c3 = self.relu(self.bn128(self.conv3_1(p2)))
        c3 = self.relu(self.bn64(self.conv3_2(c3)))
        u8 = self.relu(self.upsample(c3))
        u8 = self.relu(torch.cat((u8, c2), dim=1))
        c8 = self.relu(self.bn64(self.conv4_1(u8)))
        c8 = self.relu(self.bn32(self.conv4_2(c8)))
        u9 = self.relu(self.upsample(c8))
        u9 = self.relu(torch.cat((u9, c1), dim=1))
        c9 = self.relu(self.bn32(self.conv5_1(u9)))
        c9 = self.relu(self.bn32(self.conv5_2(c9)))
        c10 = self.relu(self.conv6(c9))

        return c10