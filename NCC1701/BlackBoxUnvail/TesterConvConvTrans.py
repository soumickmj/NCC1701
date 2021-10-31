import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn

from BlackBoxUnvail.OneLayerTester import OneLayerNetPool, OneLayerNetUnPool

mat = sio.loadmat(r"D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice51\ds2-image-test\0.mat")
k = mat['under']
k = np.expand_dims(k, 0)
kT = torch.from_numpy(k).float()

conv = nn.Conv2d(1, 1, kernel_size = 3, stride=1, padding=1)
deconv = nn.ConvTranspose2d(1,1,kernel_size=3,stride=1,padding=1,output_padding=0)

k_out_conv = conv(kT)
k_out_deconv = deconv(k_out_conv)

#poolnet = OneLayerNetPool()
#k_out_pool, k_out_pool_ind = poolnet(kT)

#unpoolnet = OneLayerNetUnPool()
#k_out_unpool = unpoolnet(k_out_pool, k_out_pool_ind)

sio.savemat(r"D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice51\ds2-image-test\0p.mat",{'k_out_pool': k_out_conv.numpy(), 'k_out_unpool': k_out_deconv.numpy()})

print('The End')