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

missingMaskDown been calculated based on missingMask, in the constructor of the ResNet class. missingMaskDown is supplied inside the ResidualBlock constructor
All the other layers, including residual blocks have data consistancy been performed.

This is inpired by AdvResnet2Dv2, just this time its been replaceced with linear layers.
"""

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable



__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ResidualBlock(nn.Module):
    def __init__(self, in_features, w_size, h_size,  missingMask):
        super(ResidualBlock, self).__init__()

        self.missingMask = missingMask
        actual_features = in_features*w_size*h_size

        heart_block = [     nn.Linear(actual_features, actual_features),
                            nn.InstanceNorm1d(actual_features),
                            nn.PReLU(num_parameters = 1, init = 0.25), #number of alphas to train - it can be either 1 or same as no of input channels.
                            nn.Dropout2d(p=0.2),
                            nn.Linear(actual_features, actual_features),
                            nn.InstanceNorm1d(actual_features) ]


        self.heart_block = nn.Sequential(*heart_block)

    def forward(self, x):
        under = x
        out = self.heart_block(x)
        out = out * self.missingMask #Missing K-Space
        out = out + under #Full k-Space
        return x + out

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=14, starting_nfeatures=1, updown_blocks=2, underDownStrategy='cropcenter', w_size=256, h_size=256, missingMask=None): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        self.missingMask = self.imgFlatten(torch.from_numpy(missingMask).float()).cuda(0) 
        #self.missingMask = self.imgFlatten(nn.functional.interpolate(self.missingMask.expand(1,1,-1,-1), (128,128))) 
        self.missingMask1 = self.missingMask.cuda(1)
        self.missingMask4 = self.missingMask.cuda(4)
        self.missingMask5 = self.missingMask.cuda(5)
        if(underDownStrategy=='cropcenter'): #27% varden undersample becomes 30% - preserves the image
            self.missingMaskDown = Variable(torch.from_numpy(self.imgFlatten(self.downCropCenter(missingMask, updown_blocks))).float()).cuda(0)
        elif(underDownStrategy=='skiplines'): #27% varden undersample becomes 28%
            self.missingMaskDown = Variable(torch.from_numpy(self.imgFlatten(self.downSkipLines(missingMask, updown_blocks))).float()).cuda(0)
        elif(underDownStrategy=='interpolate'): #27% varden undersample becomes 45%
            self.missingMaskDown = Variable(self.imgFlatten(self.downInterpolate(torch.from_numpy(missingMask).float().expand(1,1,-1,-1), updown_blocks))).cuda(0)

        self.missingMaskDown2 = self.missingMaskDown.cuda(2)
        self.missingMaskDown3 = self.missingMaskDown.cuda(3)

        in_channels = n_channels 
        out_channels = n_channels

        self.underDownStrategy = underDownStrategy

        #used for calculating how many times under sampled data to be repeated on channel dim for adding it to the result
        self.initialUnderRepeat = starting_nfeatures//2 #this calculates repeatation for original size under : after initial conv and before final conv
        if(self.initialUnderRepeat == 0):
            self.initialUnderRepeat = 1
        actual_features_starting = starting_nfeatures*w_size*h_size

        # Initial convolution block
        #intialConv = [  nn.Linear(in_channels*w_size*h_size, actual_features_starting),
        #                nn.InstanceNorm1d(actual_features_starting),
        #                nn.PReLU(num_parameters = actual_features_starting, init = 0.25) ]
        
        intialConv = [  nn.Linear(w_size*h_size, actual_features_starting),
                        nn.InstanceNorm1d(actual_features_starting),
                        nn.PReLU(num_parameters = 1, init = 0.25) ]

        # Downsampling
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2        
        for _ in range(updown_blocks):
            actual_features_in = in_features*w_size*h_size
            w_size = w_size//2
            h_size = h_size//2
            actual_features_out = out_features*w_size*h_size
            downsam += [  nn.Linear(actual_features_in, actual_features_out),
                        nn.InstanceNorm1d(actual_features_out),
                        nn.PReLU(num_parameters = 1, init = 0.25) ]
            in_features = out_features
            out_features = in_features*2
        
        self.middleUnderRepeat = in_features//2 #this calculates repeatation for original size under : after down conv and after resblocks
        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            resblocks += [ResidualBlock(in_features, w_size, h_size, self.missingMaskDown3)]
            #resblocks += [ResidualBlock(in_features)]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            actual_features_in = in_features*w_size*h_size
            w_size = w_size*2
            h_size = h_size*2
            actual_features_out = out_features*w_size*h_size
            upsam += [  nn.Linear(actual_features_in, actual_features_out),
                        nn.InstanceNorm1d(actual_features_out),
                        nn.PReLU(num_parameters = 1, init = 0.25) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer - final conv
        #finalconv = [  nn.Linear(actual_features_starting, out_channels*w_size*h_size),
        #               nn.Tanh() ]
        
        finalconv = [  nn.Linear(actual_features_starting, w_size*h_size),
                       nn.Tanh() ]

        self.intialConv = nn.Sequential(*intialConv).cuda(1)
        self.downsam = nn.Sequential(*downsam).cuda(2)
        self.resblocks = nn.Sequential(*resblocks).cuda(3)
        self.upsam = nn.Sequential(*upsam).cuda(4)
        self.finalconv = nn.Sequential(*finalconv).cuda(5)

        self.updown_blocks = updown_blocks

        print('initialization done')

    def downInterpolate(self, data, factor, algo='nearest'):
        return nn.functional.interpolate(data, (data.shape[-2]//factor,data.shape[-1]//factor), mode=algo) 

    def downCropCenter(self, data, factor):
        sideLast = data.shape[-1]//factor
        side2ndLast = data.shape[-2]//factor
        sideLastRemove = (data.shape[-1] - sideLast) // 2
        side2ndLastRemove = (data.shape[-2] - side2ndLast) // 2
        return data[...,sideLastRemove:sideLastRemove+sideLast,side2ndLastRemove:side2ndLastRemove+side2ndLast]

    def downSkipLines(self, data, factor):
        return data[...,::factor,::factor]

    def getMissingMask(self, data):
        device=data.device
        data = data.cpu().detach().numpy()
        data = data[:,0,...] + 1j*data[:,1,...]
        data = np.expand_dims(data, 1)
        nnz = data.any(axis=-1) 
        m = (np.ones(data.shape) * np.expand_dims(nnz,-1)).astype(bool)
        missingMask = (~m).astype(int)
        return torch.from_numpy(missingMask).float().to(device)

    def imgFlatten(self, data):
        #this flattens only the last two dims - works with both numpy as torch
        img_dims = (data.shape[-2] * data.shape[-1], )
        nonimg_dims = data.shape[:-2]
        return data.reshape(nonimg_dims+img_dims)

    def imgUnFlatten(self, data):
        #this unflattens only the last two dims - works with both numpy as torch - works with the principle that widht and height are same
        img_dims = (int(math.sqrt(data.shape[-1])),int(math.sqrt(data.shape[-1])))
        nonimg_dims = data.shape[:-1]
        return data.reshape(nonimg_dims+img_dims)

    def forward(self, x):
        #print('forward started')
        #x = nn.functional.interpolate(x, (128,128)) 
        if(self.underDownStrategy=='cropcenter'):
            underDown = self.downCropCenter(x, self.updown_blocks)
        elif(self.underDownStrategy=='skiplines'):
            underDown = self.downSkipLines(x, self.updown_blocks)
        elif(self.underDownStrategy=='interpolate'):
            underDown = self.downInterpolate(x, self.updown_blocks)

        #print('under down done')
        under = self.imgFlatten(x)
        under1 = under.cuda(1)
        under4 = under.cuda(4)
        under5 = under.cuda(5)
        underDown = self.imgFlatten(underDown)
        underDown2 = underDown.cuda(2)
        underDown3 = underDown.cuda(3)
        #print('cuda shift done')

        #under: cuda 1, 4, 5
        #missing mask: 1, 4, 5
        #underDown: 2, 3
        #missingDown: 2, 3

        x = self.intialConv(under1) #256
        x = x * self.missingMask1
        x = x + under1
        #print('stage 1')

        x = x.cuda(2)
        x = self.downsam(x) #64
        x = x * self.missingMaskDown2
        x = x + underDown2
        #print('stage 2')

        x = x.cuda(3)
        x = self.resblocks(x) #64
        x = x * self.missingMaskDown3
        x = x + underDown3
        #print('stage 3')

        x = x.cuda(4)
        x = self.upsam(x) #256
        x = x * self.missingMask4
        x = x + under4
        #print('stage 4')

        x = x.cuda(5)
        x = self.finalconv(x) #256
        x = x * self.missingMask5
        x = x + under5
        #print('stage 5')

        x = x.cuda(0)
        x = self.imgUnFlatten(x)
        #x = nn.functional.interpolate(x, (256,256)) 
        #print('stage final done')

        return x