#!/usr/bin/env python

"""
This is a BottleNeckNet, to be used with fourier data
"""

import torch
import torch.nn as nn
from model.kSpace.DenseNet.BottleNeck import BottleNeckNet
from model.ResNet.Resnet2Dv2b14 import ResNet
from Math.FrequencyTransformsTorch import rfft2c, irfft2c

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Not tested"

class ImgKSPNet(nn.Module):
    """description of class"""

    def __init__(self, n_channels, underMask=None, input_size=256, ksp_net=BottleNeckNet, img_net=ResNet, ksp_bottle_neck_size=64, ksp_model_type='simplelinear', ksp_activation='prelu', ksp_normalization='instance'):
        super().__init__()

        self.ksp_net = BottleNeckNet(input_size, ksp_bottle_neck_size, ksp_model_type, ksp_activation, ksp_normalization)
        self.img_net = ResNet(n_channels, final_out_sigmoid=False)

        if torch.cuda.device_count() > 1:
            print('multi')
            self.img_net.cuda(1)
            self.ksp_net.cuda(2)
            self.splitgpu = True
        else:
            self.img_net.cuda(0)
            self.ksp_net.cuda(0)
            self.splitgpu = False

        self.underMask = underMask
        if self.underMask is not None:
            self.missingMask = (~(underMask.byte())).float()

        self.finalactivation = nn.Sigmoid().cuda(0)

    def forward(self, x):     
        ##Following is the actual implementation
        #if self.training:
        #    fully_k = rfft2c(x) #supplied x is fully
        #    under_k = fully_k*self.underMask
        #    under = irfft2c(under_k)
        #else:
        #    under = x #supplied x is under
        #    under_k = rfft2c(under)
        
        ##Following is the implementation for testing the approach
        fully_k = rfft2c(x, False) #supplied x is fully
        under_k = fully_k*self.underMask
        under = irfft2c(under_k)

        #Operations in the Image Space and obtain the kSpace
        u = under.cuda(1)
        oi = self.img_net(u)
        out_img_net = oi.cuda(0)
        out_img_net_k = rfft2c(out_img_net, False)

        #Operations in the kSpace
        if self.splitgpu:
            uk = under_k.cuda(2)
            ok = self.ksp_net(uk)
            out_ksp_net_k = ok.cuda(0)
        else:
            out_ksp_net_k = self.ksp_net(under_k)

        #Combined operations
        out_combined_k = out_img_net_k + out_ksp_net_k
        missing_k = out_combined_k*self.missingMask
        final_k = missing_k + under_k
        final = irfft2c(final_k)
        return self.finalactivation(final)
