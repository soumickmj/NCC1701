#!/usr/bin/env python

"""
ResNeXt Based Image Translator
Uses 2D Images
Used in Cycle GAN
Reference: https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
        and https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
Paper: http://arxiv.org/abs/1611.05431
With static hyper parameters

Starting code is from Resnet2DSig
initial_out_features, which is the number of output features of the initial convolution block is now parameterized default value is set to 32 (in ResNet it was 64)
number of residual blocks doesn't exist anymore
instead of having n number of residual blocks, it now contains n stages of ResNeXt blocks
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


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


class ResNeXt(nn.Module):
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

    def __init__(self, n_channels=1, cardinality=8, depth=29, base_width=64, widen_factor=4, initial_out_features=32, n_resnext_stages=3):
        super(ResNeXt, self).__init__()

        in_channels = n_channels
        out_channels = n_channels

        self.cardinality = cardinality
        self.depth = depth
        #self.block_depth = (self.depth - 2) // 9 #9, becasue 3 stages, in each stage each block contains 3 convs. 2, because 1 conv before all the blocks, and 1 final layer
        self.block_depth = self.depth // (3*n_resnext_stages) #9, becasue 3 stages, in each stage each block contains 3 convs. 2, because 1 conv before all the blocks, and 1 final layer
        self.base_width = base_width
        self.widen_factor = widen_factor
        

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, initial_out_features, 7),
                    nn.InstanceNorm2d(initial_out_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling - 2 blocks
        in_features = initial_out_features
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        
        #compute input output of each stage
        self.stages = [in_features]
        delta = 1
        for _ in range(n_resnext_stages):
            widen = delta * self.widen_factor
            self.stages.append(in_features * widen)
            delta*=2

        # Residual blocks
        for i in range(n_resnext_stages):
            stage_name = 'stage_'+str(i+1)
            model += self.block(stage_name, self.stages[i], self.stages[i+1], 1)

        # Upsampling - 2 blocks
        in_features = self.stages[-1]
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, out_channels, 7),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
