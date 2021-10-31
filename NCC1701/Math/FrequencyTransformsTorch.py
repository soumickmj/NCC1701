#!/usr/bin/env python

"""
This model deals in transform to and from frequency domain.
They are all been centered (fftshit) - because of our MRI applications


Jobs:-
    1. Standard Fourier Transform (2D)
    2. Real Fourier Transform (2D)
    3. ifftshit and fftshift for PyTorch tensors
    4. roll function for pytorch tensors similar to np.roll
"""

import torch

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished, but more might be added later"

def fft2c(tensor, normalize=True):
    tensor = ifftshift(tensor, (-3, -2))
    tensor_k = torch.fft(tensor, signal_ndim=2, normalized=True)
    tensor_k = fftshift(tensor_k, (-3, -2))
    if normalize:
        temp = tensor_k.clone()**2
        tensor_k_abs = (temp[...,0]+temp[...,1]).sqrt() 
        tensor_k_unnorm = tensor_k.clone()
        for i in range(tensor_k_abs.size(0)):
            item_abs_max = tensor_k_abs[i,...].max()
            tensor_k[i,...] = tensor_k_unnorm[i,...] / item_abs_max
    return tensor_k.transpose(1,-1).squeeze() #To convert the last dim (real and imag) to channel dim

def ifft2c(tensor_k):
    tensor_k = tensor_k.unsqueeze(-1).transpose(-1,1) #To convert the channel dim to the last dim (real and imag)
    tensor_k = ifftshift(tensor_k, (-3, -2))
    tensor = torch.ifft(tensor_k, signal_ndim=2, normalized=True)
    tensor = fftshift(tensor, (-3, -2))
    return tensor

def rfft2c(tensor, normalize=True):
    tensor = ifftshift(tensor, (-2, -1))
    tensor_k = torch.rfft(tensor, signal_ndim=2, normalized=True, onesided=False)
    tensor_k = fftshift(tensor_k, (-3, -2))
    if normalize:
        temp = tensor_k.clone()**2
        tensor_k_abs = (temp[...,0]+temp[...,1]).sqrt() 
        tensor_k_unnorm = tensor_k.clone()
        for i in range(tensor_k_abs.size(0)):
            item_abs_max = tensor_k_abs[i,...].max()
            tensor_k[i,...] = tensor_k_unnorm[i,...] / item_abs_max
    return tensor_k.transpose(1,-1).squeeze(-1) #To convert the last dim (real and imag) to channel dim

def irfft2c(tensor_k):
    tensor_k = tensor_k.unsqueeze(-1).transpose(-1,1) #To convert the channel dim to the last dim (real and imag)
    tensor_k = ifftshift(tensor_k, (-3, -2))
    tensor = torch.irfft(tensor_k, signal_ndim=2, normalized=True, onesided=False)
    tensor = fftshift(tensor, (-2, -1))
    return tensor

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)