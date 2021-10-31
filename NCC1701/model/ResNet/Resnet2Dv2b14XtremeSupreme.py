
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
#from utils.TorchAct.pelu import PELU_oneparam as PELU

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

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        #PELU(),
                        nn.Dropout2d(p=0.2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]


        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_cuda=True): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        in_channels = n_channels
        out_channels = n_channels
        # Initial convolution block
        intialConv = [   nn.ReflectionPad2d(3),
                        nn.Conv2d(in_channels, starting_nfeatures, 7),
                        nn.InstanceNorm2d(starting_nfeatures),
                        nn.ReLU(inplace=True) ]

        # Downsampling
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam.append(nn.Sequential(*[  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]))
            if is_cuda:
                downsam[-1].cuda()
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            resblocks += [ResidualBlock(in_features)]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam.append(nn.Sequential(*[  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]))
            if is_cuda:
                upsam[-1].cuda()
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Sigmoid() ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = downsam
        #self.downsam = nn.Sequential(*downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = upsam
        #self.upsam = nn.Sequential(*upsam)
        self.finalconv = nn.Sequential(*finalconv)

    def forward(self, x):
        #v5: residual of v4 (in Xtreme version) + output of the first downsam added with the output of the first upsample
        iniconv = x + self.intialConv(x) #256
        down1 = self.downsam[0](iniconv) #128
        out = self.downsam[1](down1) #64
        out = out + self.resblocks(out) #64 - the input to the residual blocks is added back to the output
        out = down1 + self.upsam[0](out) #128
        out = iniconv + self.upsam[1](out) #256
        out = self.finalconv(out) #256 - the output of the upsampling block is added back to the output of the final convolution (I don't think it's needed)
        return out #final residue, sending the initial input with the output


