#!/usr/bin/env python

"""
Architecture of original DanseNet 2D
Uses 2D Images. This is the original implimentation, doesn't support image regression, only does classification.
Paper: https://arxiv.org/abs/1608.06993
Source: https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
With static hyper parameters
To call this DenseNet: net = DenseNet3(depth=100, num_classes=10, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0)
depth: total number of layers (default: 100)
num_classes: no of output prediction classes
growth_rate: number of new channels per layer (default: 12)
reduction: compression rate in transition stage (default: 0.5)
bottleneck: Use bottleneck block (True or False)
dropRate: dropout probability (default: 0.0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
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

class DenseNet(nn.Module):
    def __init__(self, n_channels=1, depth=100, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, res_blocks=34):
        #growth_rate: number of new channels per layer (default: 12)
        #depth: total number of layers (default: 100)
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(n_channels, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        ##### Going down path
        # 1st block - size becomes half
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        trans1_outsize = int(math.floor(in_planes*reduction))
        self.trans1 = TransitionBlock(in_planes, trans1_outsize, dropRate=dropRate)
        in_planes = trans1_outsize
        # 2nd block - size becomes a quarter
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        trans2_outsize = int(math.floor(in_planes*reduction))
        self.trans2 = TransitionBlock(in_planes, trans2_outsize, dropRate=dropRate)
        in_planes = trans2_outsize
        # 3rd block - size stays a quarter
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)

        # global batch norm and relu
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        # last conv after all dense blocks
        self.convF = nn.Conv2d(in_planes, n_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # Input 
        out = self.conv1(x)

        # Going down path
        out = self.trans1(self.block1(out)) #size becomes half
        out = self.trans2(self.block2(out)) #size becomes a quarter
        out = self.block3(out)
        
        # Final
        out = self.relu(self.bn1(out))

        # Output
        out = self.convF(out)

        return out