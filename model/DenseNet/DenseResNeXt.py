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


To big for GPU
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
        self.pool = nn.MaxPool2d(2, return_indices=True) #added instead of average pool, to be able to unpool later. AvgPool as no AvgUnpool
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out, indices = self.pool(out)
        #return F.avg_pool2d(out, 2)
        return out, indices 

class TransitionTransposeBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionTransposeBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.unpool = nn.MaxUnpool2d(2)
    def forward(self, x, pool_indices):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return self.unpool(out,pool_indices) #instead of average unpool max unpool used as that is not available. torch.nn layer used instead of functional layer, because indices not avilable

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

class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)

    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, name):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        self.name = name
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class DenseResNeXt(nn.Module):
    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor, name_))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor, name_))
        return block

    def __init__(self, n_channels=1, dense_depth=100, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0,
                 cardinality=8, residual_depth=29, base_width=64, widen_factor=3, n_resnext_stages=3):
        #growth_rate: number of new channels per layer (default: 12)
        #depth: total number of layers (default: 100)
        super(DenseResNeXt, self).__init__()

        self.cardinality = cardinality
        self.residual_depth = residual_depth
        #self.block_depth = (self.depth - 2) // 9 #9, becasue 3 stages, in each stage each block contains 3 convs. 2, because 1 conv before all the blocks, and 1 final layer
        self.block_depth = self.residual_depth // (3*n_resnext_stages) #9, becasue 3 stages, in each stage each block contains 3 convs. 2, because 1 conv before all the blocks, and 1 final layer
        self.base_width = base_width
        self.widen_factor = widen_factor

        in_planes = 2 * growth_rate
        n = (dense_depth - 4) / 3
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

        #### Mid blocks
        # global batch norm and relu
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Learning
        residual_learn = []

        #compute input output of each stage
        self.stages = [in_planes]
        delta = 1
        for _ in range(n_resnext_stages):
            widen = delta * self.widen_factor
            self.stages.append(in_planes * widen)
            delta*=2

        # Residual blocks
        for i in range(n_resnext_stages):
            stage_name = 'stage_'+str(i+1)
            residual_learn += self.block(stage_name, self.stages[i], self.stages[i+1], 1)
        self.residual_learn = nn.Sequential(*residual_learn)

        ##### Going up path
        # 1st or 4th block - size stays a quarter
        self.block1T = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # 2nd block - size becomes half
        self.trans2T = TransitionTransposeBlock(in_planes, trans2_outsize, dropRate=dropRate)
        in_planes = trans2_outsize
        self.block2T = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # 3rd block - size becomes orignal
        self.trans3T = TransitionTransposeBlock(in_planes, trans1_outsize, dropRate=dropRate)
        in_planes = trans1_outsize
        self.block3T = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)

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
        out, indices1 = self.trans1(self.block1(out)) #size becomes half
        out, indices2 = self.trans2(self.block2(out)) #size becomes a quarter
        out = self.block3(out)
        
        # Mid block
        out = self.relu(self.bn1(out))
        out = self.residual_learn(out)
        out = self.relu(self.bn1(out))

        # Going up path
        out = self.block1T(out)
        out = self.block2T(self.trans2T(out, indices2)) #size becomes half
        out = self.block3T(self.trans3T(out, indices1)) #size becomes original

        # Output
        out = self.convF(out)

        return out