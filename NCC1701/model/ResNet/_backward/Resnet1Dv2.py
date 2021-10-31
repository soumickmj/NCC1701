#!/usr/bin/env python

"""
Resnet Based Image Translator
Uses 2D Images
Used in Cycle GAN
Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py
Paper: arXiv:1703.10593v5 [cs.CV] 30 Aug 2018
With static hyper parameters

This version includes Dropout over Resnet2DSig.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad1d(1),
                        nn.Conv1d(in_features, in_features, 3),
                        nn.InstanceNorm1d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Dropout1d(p=0.2),
                        nn.ReflectionPad1d(1),
                        nn.Conv1d(in_features, in_features, 3),
                        nn.InstanceNorm1d(in_features)  ]


        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=34):
        super(ResNet, self).__init__()

        in_channels = n_channels
        out_channels = n_channels
        # Initial convolution block
        model = [   nn.ReflectionPad1d(3),
                    nn.Conv1d(in_channels, 64, 7),
                    nn.InstanceNorm1d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv1d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose1d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad1d(3),
                    nn.Conv1d(64, out_channels, 7),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
