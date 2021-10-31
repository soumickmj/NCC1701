#!/usr/bin/env python

"""
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
    def __init__(self, in_features, relu, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm(in_features),
                        relu(),
                        nn.Dropout2d(p=0.2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm(in_features)  ]


        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResNet(nn.Module):
    def __init__(self, n_channels=16, res_blocks=40, is_relu_leaky=True, final_out_tanh=True, do_batchnorm=True): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        if is_relu_leaky:
            relu = nn.PReLU
        else:
            relu = nn.ReLU
        if do_batchnorm:
            print('doing batchnorm')
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d       

        model = nn.ModuleList()
        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(n_channels, relu, norm)]
                 
        
        #final activation
        if final_out_tanh:
            model += [ nn.Tanh(), ]
        else:
            model += [ relu(), ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
if __name__=='__main__':
    from torchsummary import summary
    model = ResNet().cuda()
    summary(model, (16,640,320))