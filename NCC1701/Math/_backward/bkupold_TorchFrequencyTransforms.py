#!/usr/bin/env python

"""
This model deals in transform to and from frequency domain.
They are all been centered (fftshit) - because of our MRI applications

All the functions contain a optional parameter normalize. By default, it's none.
The default normalization has the direct transforms unscaled and the inverse transforms are scaled by 1/n. 
It is possible to obtain unitary transforms by setting the keyword argument norm to "ortho" (default is None) 
so that both direct and inverse transforms will be scaled by 1/sqrt{n}.

Jobs:-
    1. Standard Fourier Transform
    2. Real Fourier Transform
    3. Hermitian Fourier Transform
    4. Hartley Transform

Required Datataypes of Function parameters:-
fixedLen: int
shape: tuple - sequence of ints
axis: int
axes: tuple - sequence of ints
shiftAxes: int or tuple
normalize: None or “ortho”
norm_with_fnorm (only for hartley): bool
use_real_fourier (only for hartley): bool

currently implemented: ifft2c, fft2c

"""

import numpy as np  
import torch
from Math.Normalizations import fnorm

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished, but more might be added later"

#### PyTorch implementations of FFTShift and iFFTShift
def fftshiftT(x):
    axes = tuple(range(len(x.shape)))
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, axes)

def ifftshiftT(x):
    axes = tuple(range(len(x.shape)))
    shift = [-(dim // 2) for dim in x.shape]
    return torch.roll(x, shift, axes)


### Standard Fourier Transform

#Standard Fourier Transform - 1D
def fftc(x, fixedLen=None, axis=-1, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=shiftAxes), n=fixedLen, axis=axis, norm=normalize), axes=shiftAxes)
    return f

def ifftc(x, fixedLen=None, axis=-1, normalize=None):
    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=shiftAxes), n=fixedLen, axis=axis, norm=normalize), axes=shiftAxes)
    return f   

#Standard Fourier Transform - 2D
#def fft2c(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
#    #f1 = np.fft.fft2(x, s=shape, axes=axes, norm=normalize) #Here is the problem. When we pass axes it gives NaN
#    #f = np.fft.ifftshift(x, axes=shiftAxes)
#    #f = np.fft.fft2(f, s=shape, axes=axes, norm=normalize)
#    #f = np.fft.fftshift(f, axes=shiftAxes)
#    f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
#    return f
# This version isn't working. Giving NaN for all elements. TODO: debug the problem

def fft2cT(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    x = x.permute(0,2,3,1)
    f = torch.empty((x.shape[0], x.shape[1], x.shape[2], 2),dtype=x.dtype)    
    if(len(x.shape) == 4):
        for i in range(x.shape[0]):
            temp = torch.cat((x[i,:,:,0].unsqueeze(2), torch.zeros(x[i,:,:,0].shape).to(x.device).unsqueeze(2)), dim=2) #zero fill the imag part
            f[i,:,:,:] = fftshiftT(torch.fft(ifftshiftT(temp), signal_ndim=2))
        f = f.permute(0,3,1,2)
    else:
        f = torch.irfft(x.squeeze(), signal_ndim=2, onesided=False)
    return f

#def ifft2c(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
#    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
#    return f
# Can't trust this version as similar fft2c isn't working

def ifft2cT(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    x = x.permute(0,2,3,1)
    f = torch.empty((x.shape[0], 1, x.shape[1], x.shape[2]),dtype=x.dtype)    
    if(len(x.shape) == 4):
        for i in range(x.shape[0]):
            temp = fftshiftT(torch.ifft(ifftshiftT(x[i,:,:,:]), signal_ndim=2))
            f[i,0,:,:] = (temp[:,:,0].pow(2) + temp[:,:,0].pow(2)).sqrt() #For calculating absolute using pythagoras
    else:
        f = torch.irfft(x.squeeze(), signal_ndim=2, onesided=False)
    return f

#Standard Fourier Transform - nD
def fftNc(x, shape=None, axes=None, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f

def ifftNc(x, shape=None, axes=None, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f

### Real Fourier Transform

#Real Fourier Transform - 1D
def rfftc(x, fixedLen=None, axis=-1, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.rfft(np.fft.ifftshift(x, axes=shiftAxes), n=fixedLen, axis=axis, norm=normalize), axes=shiftAxes)
    return f

def irfftc(x, fixedLen=None, axis=-1, normalize=None):
    f = np.fft.fftshift(np.fft.irfft(np.fft.ifftshift(x, axes=shiftAxes), n=fixedLen, axis=axis, norm=normalize), axes=shiftAxes)
    return f   

#Real Fourier Transform - 2D
def rfft2c(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    f = np.fft.fftshift(np.fft.rfft2(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f

def irfft2c(x, shape=None, axes=(0,1), shiftAxes = (0,1), normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    f = np.fft.fftshift(np.fft.irfft2(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f

#Real Fourier Transform - nD
def rfftNc(x, shape=None, axes=None, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.rfftn(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f

def irfftNc(x, shape=None, axes=None, shiftAxes = None, normalize=None):
    f = np.fft.fftshift(np.fft.irfftn(np.fft.ifftshift(x, axes=shiftAxes), s=shape, axes=axes, norm=normalize), axes=shiftAxes)
    return f


### Fourier Space to Magnitude and Phase 
def f2mp(x):
    mag = np.abs(x)
    phi = np.angle(x)
    return mag, phi

def mp2f(mag, phi):
    real = mag * np.cos(phi)
    imag = mag * np.sin(phi)
    return real + 1j * imag

### Hartley Transform

#Hartley Transform - 1D
def fhtc(x, fixedLen=None, axis=-1, shiftAxes = None, norm_with_fnorm=False, use_real_fourier=False, normalize=None):
    if(use_real_fourier):
        f = rfftc(x, fixedLen=fixedLen, axis=axis, shiftAxes = shiftAxes, normalize=normalize)
    else:
        f = fftc(x, fixedLen=fixedLen, axis=axis, shiftAxes = shiftAxes, normalize=normalize)
    if(norm_with_fnorm):
        f = fnorm(f)
    h = f.real - f.imag
    return h

def ifhtc(x, fixedLen=None, axis=-1, shiftAxes = None, norm_with_fnorm=False, use_real_fourier=False, normalize=None):
    h = fhtc(x, fixedLen=fixedLen, axis=axis, shiftAxes = shiftAxes, norm_with_fnorm=norm_with_fnorm, use_real_fourier=use_real_fourier, normalize=normalize) #Based on Hartley Transform algo x = H(H(x))
    return h 

#Hartley Transform - 2D
def fht2cT(x, shape=None, axes=(0,1), shiftAxes = (0,1), norm_with_fnorm=False, use_real_fourier=False, normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    if(use_real_fourier):
        f = rfft2c(x, shape=shape, axes=axes, shiftAxes = shiftAxes, normalize=normalize)
    else:
        f = fft2cT(x, shape=shape, axes=axes, shiftAxes = shiftAxes, normalize=normalize)
    if(norm_with_fnorm):
        f = fnorm(f)
    h = f[:,0,:,:] - f[:,1,:,:]
    h = h.unsqueeze(1)
    return h
 
def ifht2cT(x, shape=None, axes=(0,1), shiftAxes = (0,1), norm_with_fnorm=False, use_real_fourier=False, normalize=None): # originally was axes=(-2,-1), shiftAxes = None
    h = fht2cT(x, shape=shape, axes=axes, shiftAxes = shiftAxes, norm_with_fnorm=norm_with_fnorm, use_real_fourier=use_real_fourier, normalize=normalize) #Based on Hartley Transform algo x = H(H(x))
    return h 

#Hartley Transform - nD
def fhtNc(x, shape=None, axes=None, shiftAxes = None, norm_with_fnorm=False, use_real_fourier=False, normalize=None):
    if(use_real_fourier):
        f = rfftNc(x, shape=shape, axes=axes, shiftAxes = shiftAxes, normalize=normalize)
    else:
        f = fftNc(x, shape=shape, axes=axes, shiftAxes = shiftAxes, normalize=normalize)
    if(norm_with_fnorm):
        f = fnorm(f)
    h = f.real - f.imag
    return h

def ifhtNc(x, shape=None, axes=None, shiftAxes = None, norm_with_fnorm=False, normalize=None):
    h = fhtNc(x, shape=shape, axes=axes, shiftAxes = shiftAxes, norm_with_fnorm=norm_with_fnorm, use_real_fourier=use_real_fourier, normalize=normalize) #Based on Hartley Transform algo x = H(H(x))
    return h 


