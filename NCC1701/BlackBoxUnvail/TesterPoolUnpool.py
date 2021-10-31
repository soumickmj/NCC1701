import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn

from BlackBoxUnvail.OneLayerTester import OneLayerNetPool, OneLayerNetUnPool

mat = sio.loadmat(r"D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice51\ds2-image-test\0.mat")
k = mat['under']
k = np.expand_dims(k, 0)
kT = torch.from_numpy(k).float()

pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

k_out_pool, indices = pool(kT)
k_out_unpool = unpool(k_out_pool, indices)

#poolnet = OneLayerNetPool()
#k_out_pool, k_out_pool_ind = poolnet(kT)

#unpoolnet = OneLayerNetUnPool()
#k_out_unpool = unpoolnet(k_out_pool, k_out_pool_ind)

sio.savemat(r"D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice51\ds2-image-test\0p.mat",{'k_out_pool': k_out_pool.numpy(), 'k_out_unpool': k_out_unpool.numpy()})

print('The End')