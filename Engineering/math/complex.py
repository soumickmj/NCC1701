"""
The original code is from: https://raw.githubusercontent.com/facebookresearch/fastMRI/main/fastmri/math.py meant for old Pytorch complex (last dim, real image seperately)
which has been modified for new PyTorch complex, and works for both PyTorch and Numpy
"""


import numpy as np
import torch


def complex_mul(x, y):
    """
    Complex multiplication.
    """
    re = x.real * y.real - x.imag * y.imag
    im = x.real * y.imag + x.imag * y.real
    return re + 1j * im


def complex_conj(x):
    """
    Complex conjugate.
    """
    return x.real - 1j * x.imag


def complex_abs(x):
    """
    Compute the absolute value of a complex valued input tensor.
    """
    return complex_abs_sq(x).sqrt()


def complex_abs_sq(x):
    """
    Compute the squared absolute value of a complex tensor.
    """
    re = x.real ** 2
    im = x.imag ** 2
    return (re+im)

def complex_modeconverter(data, mode=0, channel_dim=True):
    # 0: complex image, 1: magnitude image, 2: real image, 3: channel-wise mag+phase [-3 to invert], 4: channel-wise real+imag [-4 to invert]
    if mode == 0:
        return data
    if mode == 1:
        return abs(data)
    elif mode == 2:
        return data.real
    elif mode == 3:
        _tmp = [abs(data), data.angle() if type(data)
                is torch.Tensor else np.angle(data)]
        return (torch.cat(_tmp) if channel_dim else torch.stack(_tmp)) if type(data) is torch.Tensor else (np.concatenate(_tmp) if channel_dim else np.stack(_tmp))
    elif mode == 4:
        _tmp = [data.real, data.imag]
        return (torch.cat(_tmp) if channel_dim else torch.stack(_tmp)) if type(data) is torch.Tensor else (np.concatenate(_tmp) if channel_dim else np.stack(_tmp))
    elif mode == -3:
        _tmp = torch.split(data, data.shape[0]//2) if type(data) is torch.Tensor else np.split(data, 2)
        if not channel_dim:
            _tmp[0] = _tmp[0].squeeze(0)
            _tmp[1] = _tmp[1].squeeze(0)
        return (_tmp[0] * (torch.cos(_tmp[1]) + 1j*torch.sin(_tmp[1]))) if type(data) is torch.Tensor else (_tmp[0] * (np.cos(_tmp[1]) + 1j*np.sin(_tmp[1])))
    elif mode == -4:
        _tmp = torch.split(data, data.shape[0]//2) if type(data) is torch.Tensor else np.split(data, 2)
        if not channel_dim:
            _tmp[0] = _tmp[0].squeeze(0)
            _tmp[1] = _tmp[1].squeeze(0)
        return _tmp[0] + 1j*_tmp[1]