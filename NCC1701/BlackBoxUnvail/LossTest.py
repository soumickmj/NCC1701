from utils.HandleNifti import FileRead3D
import torch
import numpy as np
import scipy.io as sio

from Math.FrequencyTransforms import fft2c
from Math.Normalizations import fnorm

f=FileRead3D(r"D:\Datasets\OASIS1\Registered\TrainSet\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.nii.gz")[:,:,50,:]
#u10=FileRead3D(r"D:\Datasets\OASIS1\Undersampled-Registered\20190320Fresh\Varden1D10SameMask\Train\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.nii.gz")[:,:,50,:]
#u15=FileRead3D(r"D:\Datasets\OASIS1\Undersampled-Registered\20190320Fresh\Varden1D15SameMask\Train\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.nii.gz")[:,:,50,:]
#u30=FileRead3D(r"D:\Datasets\OASIS1\Undersampled-Registered\20190320Fresh\Varden1D30SameMask\Train\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.nii.gz")[:,:,50,:]

mask10 = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden10Mask.mat")['mask']
mask15 = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden15Mask.mat")['mask']
mask30 = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat")['mask']


f=fnorm(fft2c(f))
#u10=fnorm(fft2c(u10))
#u15=fnorm(fft2c(u15))
#u30=fnorm(fft2c(u30))

u10 = ( f.transpose() * mask10.transpose() ).transpose()
u15 = ( f.transpose() * mask15.transpose() ).transpose()
u30 = ( f.transpose() * mask30.transpose() ).transpose()


f=np.concatenate((np.real(f),np.imag(f)), axis=-1)
u10=np.concatenate((np.real(u10),np.imag(u10)), axis=-1)
u15=np.concatenate((np.real(u15),np.imag(u15)), axis=-1)
u30=np.concatenate((np.real(u30),np.imag(u30)), axis=-1)


f=torch.from_numpy(np.expand_dims(f.transpose((2, 0, 1)),0)).float()
u10=torch.from_numpy(np.expand_dims(u10.transpose((2, 0, 1)),0)).float()
u15=torch.from_numpy(np.expand_dims(u15.transpose((2, 0, 1)),0)).float()
u30=torch.from_numpy(np.expand_dims(u30.transpose((2, 0, 1)),0)).float()

cs = torch.nn.CosineSimilarity(dim=1) #on dim 1 returns the mask. dim 0 also, but with noise. 
pd = torch.nn.PairwiseDistance(p=2.0) #provides a kinda inverted mask
l1 = torch.nn.L1Loss()
mse = torch.nn.MSELoss()
ctc = torch.nn.CTCLoss() #TypeError("forward() missing 2 required positional arguments: 'input_lengths' and 'target_lengths'")
nll = torch.nn.NLLLoss() #Long dtype needed
pnll = torch.nn.PoissonNLLLoss() #all 1
kld = torch.nn.KLDivLoss() #interesting. all good but in opposite order. 
mar = torch.nn.MarginRankingLoss() #TypeError("forward() missing 1 required positional argument: 'target'")
hin = torch.nn.HingeEmbeddingLoss() #all 1
sml1 = torch.nn.SmoothL1Loss()
softm = torch.nn.SoftMarginLoss() #all same
mmsofmar = torch.nn.MultiLabelSoftMarginLoss() #all same
coem = torch.nn.CosineEmbeddingLoss() #TypeError("forward() missing 1 required positional argument: 'target'")
mmar = torch.nn.MultiMarginLoss() #needs long dtype
trip = torch.nn.TripletMarginLoss() #TypeError("forward() missing 1 required positional argument: 'negative'")



print('test')

