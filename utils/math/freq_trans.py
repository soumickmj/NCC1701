import numpy as np
import torch
import torch.fft

#For compatibility with PyTorch versions < 1.8 (currently master)
if "fftshift" in dir(torch.fft):
    from torch.fft import fftshift as fftshift
    from torch.fft import ifftshift as ifftshift
else:
    from .freq_trans_helpers import fftshift as fftshift
    from .freq_trans_helpers import ifftshift as ifftshift

#########
#Fourier Transforms
#########
def fftNc_pyt(data, dim=(-2, -1), norm="ortho"): #TODO will fail to work, its real data
    data = ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    data = fftshift(data, dim=dim)
    return data

def ifftNc_pyt(data, dim=(-2, -1), norm="ortho"):
    data = ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    data = fftshift(data, dim=dim)
    return data

def fftNc_np(data, axes=(-2, -1), norm="ortho"):
    data = np.fft.ifftshift(data, axes=axes)
    data = np.fft.fftn(data, axes=axes, norm=norm)
    data = np.fft.fftshift(data, axes=axes)
    return data

def ifftNc_np(data, axes=(-2, -1), norm="ortho"):
    data = np.fft.ifftshift(data, axes=axes)
    data = np.fft.ifftn(data, axes=axes, norm=norm)
    data = np.fft.fftshift(data, axes=axes)
    return data

def fft2c(data, dim=(-2, -1), norm="ortho"):
    if type(data) is torch.Tensor:
        return fftNc_pyt(data=data, dim=dim, norm=norm)
    else:
        return fftNc_np(data=data, axes=dim, norm=norm)

def ifft2c(data, dim=(-2, -1), norm="ortho"):
    if type(data) is torch.Tensor:
        return ifftNc_pyt(data=data, dim=dim, norm=norm)
    else:
        return ifftNc_np(data=data, axes=dim, norm=norm) #TODO: handle no centering cases
#########

#########
#Normalizations
#########

def fnorm_pyt(x):
    return x/torch.abs(x).max()

def fnorm_np(x):
    return x/np.abs(x).max()

def fnorm(x):
    if type(x) is torch.Tensor:
        return fnorm_pyt(x=x)
    else:
        return fnorm_np(x=x)
#########