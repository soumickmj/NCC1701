import numpy as np
import torch

from Engineering.math.freq_trans import fftNc_pyt


def root_sum_of_squares_pyt(data, dim=0, keepdim=False):
    return torch.sqrt((data ** 2).sum(dim=dim, keepdim=keepdim))


def root_sum_of_squares_np(data, dim=0):
    return np.sqrt((data ** 2).sum(dim))


def minmax(data, channel_wise=False):
    if channel_wise:
        for i in range(data.shape[0]):
            data[i, ...] = minmax(data[i, ...])
        return data
    return (data - data.min()) / (data.max() - data.min())


def minmax_complex_np(data):
    data_oo = data - data.real.min() - 1j*data.imag.min()  # origin offsetted
    return data_oo/np.abs(data_oo).max()


def minmax_complex_pyt(data, origin_offset=False, channel_wise=False):
    if channel_wise:
        for i in range(data.shape[0]):
            data[i, ...] = minmax_complex_pyt(data[i, ...], origin_offset)
        return data
    if origin_offset:
        data = data - data.real.min() - 1j*data.imag.min()  # origin offsetted
    return data/torch.abs(data).max()


def entropy_pyt(data, eps=1e-11):  # torch.finfo(torch.float32).eps):
    data = (1+1j*1)*data / torch.sum(data)
    data = data + eps
    return (-1-1j*1)*torch.sum(data * torch.log(data))


def entropy_np(data, eps=1e-11):  # np.finfo(np.float32).eps):
    data = (1+1j*1)*data / np.sum(data)
    data = data + eps
    return (-1-1j*1)*np.sum(data * np.log(data))


def entropy_loss(ent_out, ent_gt):
    if ent_out.is_complex:
        loss = torch.square(ent_gt - ent_out)
        return torch.abs(loss) + torch.angle(loss)
        # mag = torch.square(torch.abs(ent_gt) - torch.abs(ent_out))
        # ph = torch.square(torch.angle(ent_gt) - torch.angle(ent_out))
        # return mag+ph
    else:
        return torch.square(ent_gt - ent_out)

class NormUnorm(object):
    def __init__(self, tensor: torch.Tensor = None, type = "zscore", factor = 1.0):
        if type == "zscore":
            self.mean = tensor.mean(dim=(-2, -1), keepdims=True)
            self.std = tensor.std(dim=(-2, -1), keepdims=True)
        elif type == "max":
            self.max = tensor.max()
        elif type == "minmax":
            self.min = tensor.min()
            self.max = tensor.max()
        elif type == "magmax":
            self.max = torch.abs(tensor).max()
        elif type == "offset_magmax":
            self.realmin = tensor.real.min()
            self.imagmin = tensor.imag.min()
            _tensor_centreoffset = tensor - self.realmin - 1j*self.imagmin
            self.max = torch.abs(_tensor_centreoffset).max()
        elif type == "factor":
            self.max = factor
        self.type = type

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.type == "zscore":
            return (tensor - self.mean)/self.std
        elif self.type in ["max", "magmax", "factor"]:
            return tensor / self.max
        elif self.type == "minmax":
            return (tensor - self.min) / (self.max - self.min)
        elif self.type == "offset_magmax":
            _tensor_centreoffset = tensor - self.realmin - 1j*self.imagmin
            return _tensor_centreoffset / self.max

    def unnormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.type == "zscore":
            return tensor * self.std + self.mean
        elif self.type in ["max", "magmax", "factor"]:
            return tensor * self.max
        elif self.type == "minmax":
            return (tensor * (self.max - self.min)) + self.min
        elif self.type == "offset_magmax":
            return (tensor * self.max) + self.realmin + 1j*self.imagmin