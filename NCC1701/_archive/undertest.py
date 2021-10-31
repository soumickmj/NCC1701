from utils.HandleNifti import FileSave
import scipy.io as sio
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
import utils.fastMRI.TorchDSTransforms as transforms
from Math.FrequencyTransforms import ifft2c, fft2c

data = h5py.File(r"B:\Soumick\Challange DSs\fastMRI\Brain\multicoil_train\file_brain_AXFLAIR_200_6002474.h5")
slc = data['kspace'][10]

s=slc[10]

s = fft2c(abs(ifft2c(s)))
h = s.real - s.imag


fft_input_ = np.fft.fft2(h)
dht = fft_input_.real - fft_input_.imag

def fourier2hartley(x):
    x = np.fft.ifftshift(x)
    return np.fft.fftshift(x.real - x.imag)

def hartley(x):
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    return k.real - k.imag
s = transforms.to_tensor(slc[10])
h  = transforms.fourier_to_hartley(s)

s2 = transforms.hartley_to_fourier(h)


#s = slc[10] / abs(slc[10]).max()


h = s.real - s.imag

uh = h.copy()
uh[::2] = 0

def gaussianPDF2D(slice, sigma=0.5):
    Xie1 = 5000/(2*math.pi) #Sharpness of the distribution 1, 0 <r <r0
    Xie2 = sigma*1000000/(2*math.pi) #Sharpness of distribution 2, r0 <r <r_max
    s = 0 #diagonal extension of distribution - / +
    r1 = 1 #> = 1, Determines the size of the full coverage of the k-space center, the height of the distribution 1
    r2 = 0.60 #<= 1, influence on the probabilities in the edge of the k-space, height of the distribution 2

    #Fit to mask size
    x1 = np.array(range(slice.shape[0]))
    x2 = np.array(range(slice.shape[1]))     
    X1, X2 = np.meshgrid(x1,x2) 

    #Distribution function 0 <r <r0
    mu1 = np.array([slice.shape[0]//2, slice.shape[1]//2])                         #Average
    #Sigma1 = 1/(2*math.pi)*np.array([[Xie1,Xie1*s], [Xie1*s,Xie1]])         #Covariance matrix, alternative
    Sigma1 = np.array([[Xie1,Xie1*s], [Xie1*s,Xie1]])                     # Covariance matrix
    F1 = (r1*Xie1)*2*math.pi*mvn.pdf(np.array([X1.flatten(), X2.flatten()]).transpose(),mu1,Sigma1)    # 2d Normal distribution 1

    F1 = np.reshape(F1,(len(x2),len(x1))).transpose()

    #Distribution function r0 <r <r_max
    mu2 = np.array([slice.shape[0]//2, slice.shape[1]//2])                        #Average
    #Sigma2 = 1/(2*math.pi)*np.array([[Xie2,Xie2*s], [Xie2*s,Xie2]])         #Covariance matrix, alternative
    Sigma2 = np.array([[Xie2,Xie2*s], [Xie2*s,Xie2]])                     # Covariance matrix
    F2 = (r2*Xie2)*2*math.pi*mvn.pdf(np.array([X1.flatten(), X2.flatten()]).transpose(),mu2,Sigma2)    #2d Normal distribution 2

    F2 = np.reshape(F2,(len(x2),len(x1))).transpose()

    #Overlay of both distributions
    #F = (F1 + F2)/2;
        
    PDF = F1
    idx = PDF <= F2
    PDF[idx] = F2[idx]
    PDF[PDF>=1] = 1

    return PDF

p=gaussianPDF2D(h,0.5)
p1 = 1-p
h1 = p1*h
uh1 = p1*uh
a1=np.mean(h1,axis=1)
a0=np.mean(h,axis=1)

def gaussian(x,sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def X(x,sig):
    g=gaussian(x,sig)
    #weight=g/g.sum()
    #weight = [gl/gl.sum() for gl in g]
    weight = g
    return weight

FileSave(sio.loadmat(r"E:\Datasets\ResNetTestSet\Skyra\MATs_IXIRot\MidSlices\1DVarden30Mask\0.mat")['fully'].squeeze(), r"E:\Datasets\ResNetTestSet\Skyra\MATs_IXIRot\MidSlices\1DVarden30Mask\0.nii")


import os
import numpy as np
import scipy.io as sio
from pathlib import Path
root = r'E:\Datasets\ResNetTestSet\Skyra\MATs'
destinRoot = r'E:\Datasets\ResNetTestSet\Skyra\MATs_IXIRot'
for path in Path(root).rglob('*.mat'):
    mat = sio.loadmat(path)
    mat['fully'] = np.expand_dims(np.rot90(mat['fully'].squeeze(), k=3), 0)
    mat['under'] = np.expand_dims(np.rot90(mat['under'].squeeze(), k=3), 0)
    os.makedirs(os.path.dirname(path).replace(root, destinRoot), exist_ok=True)
    sio.savemat(str(path).replace(root, destinRoot), mat)



#from pynvml import *


import h5py
import ismrmrd.xsd
import pandas as pd
from utils.RAW.ISMRMRDReader import ISMRMRDReader
from utils.MRITorchDSTransforms import CorruptByMotionNP, CorruptByMotionNPMultiCoil
import numpy as np
import nibabel as nib
from Math.FrequencyTransforms import fft2c, ifft2c, fht2c, ifht2c
from utils.HandleNifti import FileRead3D, FileSave
from utils.fastMRI import TorchDSTransforms as T

data = h5py.File(r"D:\ValTest\file_brain_AXT2_200_6002317.h5", 'r') 
basicinfo = dict(data.attrs)
nSlice, nCoil, hKSP, wKSP = data['kspace'].shape
_, hIMG, wIMG = data['reconstruction_rss'].shape
headerXML = data['ismrmrd_header'].value
headerDict = ISMRMRDReader.readHeaderFromXMLExtended(headerXML)
headerDict = {f'zISMRM_{k}': v for k, v in headerDict.items()}
basicinfo.update({'nSlice':nSlice, 'nCoil':nCoil, 'hKSP':hKSP, 'wKSP':wKSP, 'hIMG':hIMG, 'wIMG':wIMG})
headerDict.update(basicinfo)
df = pd.DataFrame.from_dict([headerDict, headerDict])

sliceKSP = data['kspace'][5]
sliceIMG = (T.complex_abs(T.ifft2(T.to_tensor(sliceKSP)))).numpy()
FileSave(np.transpose(sliceIMG, (1,2,0)), r"D:\ValTest\AXT2_200_MoCo_Original.nii")
c = CorruptByMotionNP()
c2 = CorruptByMotionNPMultiCoil()

x, r = c(sliceIMG.copy())
x2, r2 = c2(sliceIMG.copy())
FileSave(np.transpose(x2, (1,2,0)), r"D:\ValTest\AXT2_200_MoCo_Same.nii")

FileSave(np.transpose(x, (1,2,0)), r"D:\ValTest\AXT2_200_MoCo_Diff.nii")

FileSave(np.transpose(sliceIMG, (1,2,0)), r"D:\ValTest\AXT2_200_MoCo_Original2.nii")


I = np.squeeze(FileRead3D(r"D:\CloudData\OneDrive\OvGU\Study Data\Codes\OneShotImageRegistration-master\OneShotImageRegistration-master\resources\benz_CenterRatio20.nii"))

FileSave(None, r"D:\CloudData\OneDrive\OvGU\Study Data\Codes\OneShotImageRegistration-master\OneShotImageRegistration-master\resources\TwoTP_20\img0.nii.gz")
#I = np.squeeze(FileRead3D(r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\PublicDSs\OASIS1\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.img"))


#K = fft2c(I)
#I_b = ifft2c(K).astype(np.int32)
#FileSave(I_b, r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Chompunuch\4Soumick\HR07fi_Python32.nii.gz')


######nii = nib.load(r'S:\MEMoRIAL_SharedStorage_M1.2+4+7\Chompunuch\4Soumick\HR07fi.nii.gz')
######data = nii.get_data()
######nii.set_data(data)
######nib.save(nii, file_path)

####import scipy.io as sio
####mat_contents = sio.loadmat('1DVardenMask.mat')
####mask = mat_contents['mask']

####import matplotlib.pyplot as plt

####plt.figure(1)
####imgplot = plt.imshow(mask)

####import numpy as np
####invmask=(~mask.astype(bool)).astype("float64")
####imgplot2 = plt.imshow(invmask)

####print('s')


##### new code for kspace normalization

###from utils.HandleNifti import FileRead3D, FileSave
###import numpy as np
###import matplotlib.pyplot as plt
###import math
###from Math.FrequencyTransforms import fft2c, ifft2c, fht2c, ifht2c
###from Math.Normalizations import fnorm, hnorm

###vol_mask = FileRead3D(r"D:\New folder\MR_data_batch1\MR_data_batch1_X\Patient_1\T1DUAL_mask.nii")

###img = FileRead3D(r"D:\Temp\vol\OAS1_0001_MR1\OAS1_0001_MR1_mpr-1_anon.img")
###k = fft2c(img)
###img = img[:,:,50,0]
###k = fft2c(img)
###h = fht2c(img)
###hn = fht2c(img, norm_with_fnorm=True) 

####kspace norm
###k_ = fnorm(k)
###img_ = ifft2c(k_)
###FileSave(img_, r"D:\Temp\vol\OAS1_0001_MR1\K.nii.gz")

####hartley space norm
###h_ = hnorm(h)
###imgH_ = ifht2c(h_)
###FileSave(imgH_, r"D:\Temp\vol\OAS1_0001_MR1\H.nii.gz")

###im = ifft2c(k)
###FileSave(im, r"D:\Temp\vol\OAS1_0001_MR1\I.nii.gz")

####hartley space already norm using ksapce
###imgHn = ifht2c(hn)
###FileSave(imgHn, r"D:\Temp\vol\OAS1_0001_MR1\Hn.nii.gz")

###f = np.fft.fft2(img)
###fshift = np.fft.fftshift(f)


###magnitude_spectrum = np.abs(fshift)
###phase_spectrum = (np.angle(fshift))

####################################################
#### Normalize Data ################################
####################################################
###def normalize_complex_arr(a):
###    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
###    return a_oo/np.abs(a_oo).max()
    
###norm_kspace = normalize_complex_arr(f)  

###norm_kspace_ = np.fft.fftshift(norm_kspace)
###mag_spec_scaled = np.log(np.abs(norm_kspace_))
###pha_spec_scaled = (np.angle(norm_kspace_))

###img_ = np.fft.ifft2(norm_kspace)

###FileSave(img_, r"D:\Temp\vol\OAS1_0001_MR1\K.nii.gz")

###ks = mag_spec_scaled * math.exp(pha_spec_scaled*1j)

###print("hi")

##import scipy.io as sio
##from utils.MRITorchDSTransforms import CorruptByMotion
##import matplotlib.pyplot as plt
##c = CorruptByMotion()
##v = sio.loadmat(r"E:\Datasets\MATs\20190320Fresh\OASIS-AllCombined4-Slice50\ds1-image-train\0.mat")['fully'].squeeze()

##i = c(v)
##plt.imshow(i)
##plt.show()



print('ss')