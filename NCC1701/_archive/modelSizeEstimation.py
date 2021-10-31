#!/usr/bin/env python

"""
This module helps to estimate size of the given Convolutional Translator
"""

import numpy as np
#from model.UNet.UnetMRIPaper import UNet as netModel
#from model.ResNeXt.ResNeXt2D import ResNeXt as netModel
#from model.UNet.UnetMRIPaper import UNet as netModel
#from model.ResNet.Resnet2Dv2b14 import ResNet as netModel
#from model.ReferenceModels.RAKI import RAKI as netModel
#from model.kSpace.CUNet import CUNet as netModel
#from model.kSpace.CInvUNet import CInvUNet as netModel
#from model.ResNet.Resnet3Dv2b14_4 import ResNet as netModel
from model.VAE.ConvVAE import ConvVAE as netModel
#from model.kSpace.DenseNet.BottleNeck import BottleNeckNet as netModel
#from model.kSpace.ImgKSPNet import ImgKSPNet as netModel
#from model.kSpace.ResNet.Resnet2Dv2b14_ksv2 import ResNet as netModel
#from model.kSpace.ResNet.AdvResnet2Dv2 import ResNet as netModel
#from model.kSpace.ResNet.LinearResnet2Dv2p2_ReducedIN import ResNet as netModel
#from model.UNet.DenseUNet2Dv1 import DenseUNet as netModel
from Math.PyTorchModelSize import SizeEstimator
from Math.PyTorchModelSummary import ModelSummary
from Math.torch_receptive_field import receptive_field
import torch

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished, but not tested"

batch_size = 1
no_of_channels = 1
image_dim = (10,32,32) #height: 256, width: 256, no of slices: 128
image_dim = (128,128) #height: 256, width: 256, no of slices: 128
input_shape = (no_of_channels, ) + image_dim
#input_shape = (batch_size, ) + image_shape

#se = SizeEstimator(netModel, input_size=input_shape,flattened_layer=[-1])
##flattened_layer=[-1] because there is no flatten operation in forward method
#print('Estimated Size (MB, bits): ' + str(se.estimate_size()))
## Returns
## (size in megabytes, size in bits)
## (408.2833251953125, 3424928768)
#print('Parameter Bits: ' + str(se.param_bits)) # bits taken up by parameters
#print('Bits stored for Forward and Backward: ' + str(se.forward_backward_bits)) # bits stored for forward and backward
#print('Bits for Input: ' + str(se.input_bits)) # bits for input

import scipy.io as sio
#mat_contents = sio.loadmat(r"/home/schatter/Code/EnterpriseV1/Gen2/v3-20190522/1DVarden30Mask.mat")
#mat_contents = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat")
#mask = mat_contents['mask']
#mask = mask[112:112+32,112:112+32]
mask = np.zeros((256,256,256))

#netModelObj = netModel(no_of_channels, missingMask=mask)
#netModelObj = netModel(no_of_channels, underMask=torch.from_numpy(mask).float())
#netModelObj = netModel(32,0)
#netModelObj = netModel(1,0)
#netModelObj.train()
netModelObj = netModel(no_of_channels)
#import torch
#torch.save(netModelObj, 's4.pth')
summary = ModelSummary(netModelObj, input_shape, 4, batch_size, 'cpu')
summary.getSummary()

#receptive_field(netModelObj, input_size=input_shape, device="cpu")\
