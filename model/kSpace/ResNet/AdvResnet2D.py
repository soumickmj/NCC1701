#!/usr/bin/env python

"""
Resnet Based Image Translator
Uses 2D Images
Used in Cycle GAN
Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py
Paper: arXiv:1703.10593v5 [cs.CV] 30 Aug 2018
With static hyper parameters

This version is the version submitted for MICAI 2019, but tailored for k-Space.
All ReLU been replaced by PReLU. This uses PReLU with number of parameters equals to the input no of features.
Final Sigmoid been replaced by, TanH.

This is a modified version over what just been discussed. Here the k-Space from the input is preserved till the output.
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        #self.missingMask = missingMask

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.PReLU(num_parameters = in_features, init = 0.25), #number of alphas to train - it can be either 1 or same as no of input channels.
                        nn.Dropout2d(p=0.2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]


        self.conv_block = nn.Sequential(*conv_block)

    def getMissingMask(self, data):
        device=data.device
        data = data.cpu().detach().numpy()
        nnz = data.any(axis=-1) 
        m = (np.ones(data.shape) * np.expand_dims(nnz,-1)).astype(bool)
        missingMask = (~m).astype(int)
        return torch.from_numpy(missingMask).float().to(device)

    def forward(self, x):
        missingMask = self.getMissingMask(x)
        out = self.conv_block(x)
        out = out * missingMask #Missing K-Space
        x = out + x #Full k-Space
        return x 

    #def forward(self, *args):
    #    x = args[0]
    #    #under = args[1]
    #    under = x
    #    out = self.conv_block(x)
    #    out = out * self.missingMask #Missing K-Space
    #    out = out + under #Full k-Space
    #    return x + out

class LambdaHandler():
    def initializer(self, updown_blocks, underDownStrategy):
        if(underDownStrategy == 'interpolate'): #27% varden undersample becomes 45%
            self.performDownUnder = lambda x: self.downInterpolate(x, updown_blocks*2, algo='bilinear')
        elif(underDownStrategy == 'cropcenter'): #27% varden undersample becomes 30% - preserves the image
            self.performDownUnder = lambda x: self.downCropCenter(x, updown_blocks*2)
        elif(underDownStrategy == 'skiplines'): #27% varden undersample becomes 28%
            self.performDownUnder = lambda x: self.downSkipLines(x, updown_blocks*2)

    def downInterpolate(self, data, factor, algo='bilinear'):
        return nn.functional.interpolate(data, (data.shape[-2]//factor,data.shape[-1]//factor), mode=algo) 

    def downCropCenter(self, data, factor):
        sideLast = data.shape[-1]//factor
        side2ndLast = data.shape[-2]//factor
        sideLastRemove = (data.shape[-1] - sideLast) // 2
        side2ndLastRemove = (data.shape[-2] - side2ndLast) // 2
        return data[...,sideLastRemove:sideLastRemove+sideLast,side2ndLastRemove:side2ndLastRemove+side2ndLast]

    def downSkipLines(self, data, factor):
        return under[...,::factor,::factor]

lh=LambdaHandler()

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, underDownStrategy='cropcenter'): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        in_channels = n_channels
        out_channels = n_channels

        #used for calculating how many times under sampled data to be repeated on channel dim for adding it to the result
        self.initialUnderRepeat = starting_nfeatures//2 #this calculates repeatation for original size under : after initial conv and before final conv

        # Initial convolution block
        intialConv = [   nn.ReflectionPad2d(3),
                        nn.Conv2d(in_channels, starting_nfeatures, 7),
                        nn.InstanceNorm2d(starting_nfeatures),
                        nn.PReLU(num_parameters = starting_nfeatures, init = 0.25) ]

        # Downsampling
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2        
        for _ in range(updown_blocks):
            downsam += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.PReLU(num_parameters = out_features, init = 0.25) ]
            in_features = out_features
            out_features = in_features*2
        
        self.middleUnderRepeat = in_features//2 #this calculates repeatation for original size under : after down conv and after resblocks
        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            #resblocks += [ResidualBlock(in_features, self.missingMaskDown)]
            resblocks += [ResidualBlock(in_features)]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.PReLU(num_parameters = out_features, init = 0.25) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer - final conv
        finalconv = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, out_channels, 7),
                        nn.Tanh() ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.Sequential(*downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.Sequential(*upsam)
        self.finalconv = nn.Sequential(*finalconv)

        self.updown_blocks = updown_blocks

        ##For downsample the under
        #if(underDownStrategy == 'interpolate'): #27% varden undersample becomes 45%
        #    self.performDownUnder = lambda x: self.downInterpolate(x, updown_blocks*2, algo='bilinear')
        #elif(underDownStrategy == 'cropcenter'): #27% varden undersample becomes 30% - preserves the image
        #    self.performDownUnder = lambda x: self.downCropCenter(x, updown_blocks*2)
        #elif(underDownStrategy == 'skiplines'): #27% varden undersample becomes 28%
        #    self.performDownUnder = lambda x: self.downSkipLines(x, updown_blocks*2)
        #lh.initializer(updown_blocks, underDownStrategy)
        #self.performDownUnder = lh.performDownUnder

    def downInterpolate(self, data, factor, algo='bilinear'):
        return nn.functional.interpolate(data, (data.shape[-2]//factor,data.shape[-1]//factor), mode=algo) 

    def downCropCenter(self, data, factor):
        sideLast = data.shape[-1]//factor
        side2ndLast = data.shape[-2]//factor
        sideLastRemove = (data.shape[-1] - sideLast) // 2
        side2ndLastRemove = (data.shape[-2] - side2ndLast) // 2
        return data[...,sideLastRemove:sideLastRemove+sideLast,side2ndLastRemove:side2ndLastRemove+side2ndLast]

    def downSkipLines(self, data, factor):
        return under[...,::factor,::factor]

    def getMissingMask(self, data):
        device=data.device
        data = data.cpu().detach().numpy()
        data = data[:,0,...] + 1j*data[:,1,...]
        data = np.expand_dims(data, 1)
        nnz = data.any(axis=-1) 
        m = (np.ones(data.shape) * np.expand_dims(nnz,-1)).astype(bool)
        missingMask = (~m).astype(int)
        return torch.from_numpy(missingMask).float().to(device)
    
    def getMissingMaskTorch(self, data):
        device=data.device
        data = data[:,0,:,:] 
        data = data.unsqueeze(1)
        nnz = data.any(dim=-1) 
        m = (np.ones(data.shape) * np.expand_dims(nnz,-1)).astype(bool)
        missingMask = (~m).astype(int)
        return torch.from_numpy(missingMask).float().to(device)

    def forward(self, x):
        under = x
        #underDown = self.performDownUnder(under)
        underDown = self.downCropCenter(under, self.updown_blocks*2)
        missingMask = self.getMissingMaskTorch(under)
        missingMaskDown = self.getMissingMask(underDown)

        x = self.intialConv(x) #256
        x = x * missingMask
        x = x + under.repeat_interleave(self.initialUnderRepeat, 1)

        x = self.downsam(x) #64
        x = x * missingMaskDown
        x = x + underDown.repeat_interleave(self.middleUnderRepeat, 1)

        x = self.resblocks(x) #64
        x = x * missingMaskDown
        x = x + underDown.repeat_interleave(self.middleUnderRepeat, 1)

        x = self.upsam(x) #256
        x = x * missingMask
        x = x + under.repeat_interleave(self.initialUnderRepeat, 1)

        x = self.finalconv(x) #256
        x = x * missingMask
        x = x + under

        return x