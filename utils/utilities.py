import torch
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import torchcomplex.nn.functional as cF
from utils.signaltools import resample

class Interpolator():
    def __init__(self, mode=None):
        if mode in ["sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]:
            self.mode = mode
        else:
            self.mode = None

    def perform_sinc(self, images, out_shape):
        axes = np.argwhere(np.equal(images.shape[2:], out_shape) == False).squeeze(1) #2 dims for batch and channel
        out_shape = [out_shape[i] for i in axes]
        return resample(images, out_shape, axis=axes+2) #2 dims for batch and channel

    def __call__(self, images, out_shape):
        if self.mode is None:
            return images
        elif images.is_complex():
            return cF.interpolate(images, size=out_shape, mode=self.mode)
        elif self.mode == "sinc":
            return self.perform_sinc(images, out_shape)
        else:
            return F.interpolate(images, size=out_shape, mode=self.mode)