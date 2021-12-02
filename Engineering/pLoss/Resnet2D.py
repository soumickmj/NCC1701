#!/usr/bin/env python

"""
Original file Resnet2Dv2b14 of NCC1701
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.TorchAct.pelu import PELU_oneparam as PELU

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class ResidualBlock(nn.Module):
    def __init__(self, in_features, relu, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm(in_features),
                      relu(),
                      nn.Dropout2d(p=0.2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResNet(nn.Module):
    # should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_n_features=64, updown_blocks=2, is_relu_leaky=True, final_out_sigmoid=True, do_batchnorm=True):
        super(ResNet, self).__init__()

        if is_relu_leaky:
            relu = nn.PReLU
        else:
            relu = nn.ReLU
        if do_batchnorm:
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, starting_n_features, 7),
                 norm(starting_n_features),
                 relu()]

        # Downsampling
        in_features = starting_n_features
        out_features = in_features*2
        for _ in range(updown_blocks):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      norm(out_features),
                      relu()]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features, relu, norm)]

        # Upsampling
        out_features = in_features//2
        for _ in range(updown_blocks):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      norm(out_features),
                      relu()]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(starting_n_features, out_channels, 7)]

        # final activation
        if final_out_sigmoid:
            model += [nn.Sigmoid(), ]
        else:
            model += [relu(), ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
