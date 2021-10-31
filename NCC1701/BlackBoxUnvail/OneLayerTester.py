#!/usr/bin/env python

"""
This module is to test just output of one layer
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "WIP"

class OneLayerNetPool(nn.Module):
    def __init__(self):
        super(OneLayerNetPool, self).__init__()

        self.layer = nn.AdaptiveAvgPool2d(128)

    def forward(self, x):
        out = self.layer(x)
        return out

class OneLayerNetUnPool(nn.Module):
    def __init__(self):
        super(OneLayerNetUnPool, self).__init__()

        self.layer = nn.MaxUnpool2d(2)

    def forward(self, x, indices):
        out = self.layer(x, indices)
        return out

class OneLayerNetUpsample(nn.Module):
    def __init__(self):
        super(OneLayerNetUpsample, self).__init__()

        self.layer = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        out = self.layer(x)
        return out

import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
from Math.FrequencyTransforms import ifft2c

mat= sio.loadmat(r"D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice51\ds1-fourier-train\0.mat")
#mat= sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat")

fully = mat['under']
#fully=np.expand_dims(fully, 0)
#fully = torch.from_numpy(np.expand_dims(fully, 0)).float()
under = fully

encoding = OneLayerNetPool()
#encoded, indices = encoding(fully)

decoding = OneLayerNetUpsample()
#decoded = decoding(encoded, indices)

#encoding = nn.Conv2d(2, 2, kernel_size = 3, stride=1, padding=1)
#decoding = nn.ConvTranspose2d(2,2,kernel_size=3,stride=1,padding=1,output_padding=0)



#encoded = encoding(fully)
#encoded = torch.nn.functional.interpolate(fully, 128, mode='bilinear')
#encoded[encoded<0.5] = 0
#encoded[encoded>=0.5] = 1
#decoded = torch.nn.functional.interpolate(encoded, 256, mode='bilinear')
#decoded[decoded<0.5] = 0
#decoded[decoded>=0.5] = 1
nnz = under.any(axis=-1) 
m = (np.ones(under.shape) * np.expand_dims(nnz,-1)).astype(bool)

under = torch.from_numpy(np.expand_dims(under, 0)).float()

underDown = nn.functional.interpolate(under, (under.shape[-2]//4,under.shape[-1]//4), mode='bilinear')
underDown = underDown.detach().numpy()
nnz = underDown.any(axis=-1) 
m = (np.ones(underDown.shape) * np.expand_dims(nnz,-1)).astype(bool)
#underDown = underDown.detach().numpy().squeeze()
#underDown = underDown[0,...] + 1j * underDown[1,...]
#underDown = ifft2c(underDown)

underDown = under[...,under.shape[-2]//8:under.shape[-2],under.shape[-1]//8:under.shape[-1]]
underDown = underDown.detach().numpy()
nnz = underDown.any(axis=-1) 
m = (np.ones(underDown.shape) * np.expand_dims(nnz,-1)).astype(bool)
#underDown = underDown.detach().numpy().squeeze()
#underDown = underDown[0,...] + 1j * underDown[1,...]
#underDown = ifft2c(underDown)


underDown = under[...,::4,::4]
underDown = underDown.detach().numpy()
nnz = underDown.any(axis=-1) 
m = (np.ones(underDown.shape) * np.expand_dims(nnz,-1)).astype(bool)
#underDown = underDown.detach().numpy().squeeze()
#underDown = underDown[0,...] + 1j * underDown[1,...]
#underDown = ifft2c(underDown)


nnz = underDown.any(axis=-1) 
m = (np.ones(underDown.shape) * np.expand_dims(nnz,-1)).astype(bool)
nnz = under.any(axis=-1) 
m = (np.ones(under.shape) * np.expand_dims(nnz,-1)).astype(bool)


from resizeimage import resizeimage
encoded = resizeimage.resize_cover(fully, [128, 128])

encoded = encoded.detach().numpy().squeeze()
decoded = decoded.detach().numpy().squeeze()
fully = fully.numpy().squeeze()

encoded = encoded[0,...] + 1j * encoded[1,...]
decoded = decoded[0,...] + 1j * decoded[1,...]
fully = fully[0,...] + 1j * fully[1,...]

encodedI = ifft2c(encoded)
decodedI = ifft2c(decoded)
fullyI = ifft2c(fully)

print('end')


#class OneLayerNetConv2D(nn.Module):
#    def __init__(self):
#        super(OneLayerNetUnPool, self).__init__()

#        self.layer = 

#    def forward(self, x, indices):
#        out = self.layer(x, indices)
#        return out

