#!/usr/bin/env python

"""
Architecture of DanseUNet 2D
Uses 2D Images
Source: https://link.springer.com/chapter/10.1007%2F978-3-030-00928-1_25
To call the danse blocks
self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
block = BasicBlock
in_planes = 2 * growth_rate
self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
where,
n = no of conv (default: 5)
in_planes = no of filters in the first layer
growth_rate = growth_rate (default: 16)
block = block object created earlier
dropRate = droprate (default: 0.0)
"""

import math
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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlockContract(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockContract, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class TransitionBlockExpand(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockExpand, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.deconv1_conv(self.deconv1_upsample(self.relu(self.bn1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseUNet(nn.Module): #Rough code currently
    def __init__(self, n_channels, n_conv=5, growth_rate=16, bottleneck=False, dropRate=0.0):
        #growth_rate: number of new channels per layer (default: 12)
        #depth: total number of layers (default: 100)
        super(DenseUNet, self).__init__()
        if bottleneck == True:
            n_conv = n_conv/2
            block = BottleneckBlock
        else:
            block = BasicBlock

        # ------------------
        #  Contracting Path
        # ------------------

        # conv before any dense block (conv1 + con2)
        self.conv_pre1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)        
        in_planes = 64

        # 1st block
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate)
        self.trans1 = TransitionBlockContract(in_planes, 64, dropRate=dropRate)
        in_planes = 64

        # 2nd block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block2 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate)
        self.trans2 = TransitionBlockContract(in_planes, 64, dropRate=dropRate)        
        in_planes = 64

        # ------------------
        #  Expanding Path
        # ------------------

        # 3rd block
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.block3 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate) 
        self.trans3 = TransitionBlockExpand(in_planes, 64, dropRate=dropRate)
        in_planes = 128 #Because of contact
        
        # 4th block        
        self.conv4 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.block4 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate) 
        self.trans4 = TransitionBlockExpand(in_planes, 64, dropRate=dropRate)
        in_planes = 64

         # 4_2th block        
        self.conv4_2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.block4_2 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate) 
        self.trans4_2 = TransitionBlockExpand(in_planes, 64, dropRate=dropRate)
        in_planes = 128

        # 5th block        
        self.conv5 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.block5 = DenseBlock(n_conv, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n_conv*growth_rate) 
        self.trans5 = TransitionBlockExpand(in_planes, n_channels, dropRate=dropRate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_conv = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n_conv))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_pre1(x)
        dense1 = self.trans1(self.block1(self.conv1(x)))
        x = F.avg_pool2d(dense1, 2)
        dense2 = self.trans2(self.block2(self.conv2(x)))
        x = F.avg_pool2d(dense2, 2)
        x = self.trans3(self.block3(self.conv3(x)))
        x = torch.cat((x, dense2), dim=1)  
        x = self.trans4(self.block4(self.conv4(x)))
        x = self.trans4_2(self.block4_2(self.conv4_2(x)))
        x = torch.cat((x, dense1), dim=1)
        x = self.trans5(self.block5(self.conv5(x)))

        return x

