from typing import Tuple, Union

import numpy as np
import torch
import torchvision
from Engineering.transforms.transforms import (ApplyOneOf, SuperTransformer,
                                               padIfNeeded)
from scipy.fftpack import ss_diff
from skimage import exposure

################# Contrast Augmentations #########################


class AdaptiveHistogramEqualization(SuperTransformer):
    def __init__(
            self,
            kernel_size: Tuple[int] = (25, 100),
            clip_limit: float = 0.01,
            nbins: int = 512,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def apply(self, inp):
        out = exposure.equalize_adapthist(inp, kernel_size=np.random.randint(
            self.kernel_size[0], high=self.kernel_size[1], size=(1))[0], clip_limit=self.clip_limit, nbins=self.nbins).astype(inp.dtype)
        return (out-out.min())/(out.max()-out.min()+np.finfo(np.float32).eps)


class AdjustGamma(SuperTransformer):
    def __init__(
            self,
            gamma: Tuple[float] = (0.75, 1.75),
            gain: float = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.gain = gain

    def apply(self, inp):
        out = exposure.adjust_gamma(inp, gamma=np.random.uniform(
            self.gamma[0], self.gamma[1], 1)[0], gain=self.gain).astype(inp.dtype)
        return (out-out.min())/(out.max()-out.min()+np.finfo(np.float32).eps)


class AdjustSigmoid(SuperTransformer):
    def __init__(
            self,
            cutoff: Tuple[float] = (0.01, 0.75),
            gain: Tuple[int] = (1, 4),
            inv: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.gain = gain
        self.inv = inv

    def apply(self, inp):
        out = exposure.adjust_sigmoid(inp, cutoff=np.random.uniform(self.cutoff[0], self.cutoff[1], 1)[0],
                                      gain=np.random.randint(
            self.gain[0], high=self.gain[1], size=(1))[0],
            inv=self.inv).astype(inp.dtype)
        return (out-out.min())/(out.max()-out.min()+np.finfo(np.float32).eps)


class AdjustLog(SuperTransformer):
    def __init__(
            self,
            gain: Tuple[float] = (-0.5, 0.5),
            inv: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.inv = inv

    def apply(self, inp):
        out = np.abs(exposure.adjust_log(inp, gain=np.random.uniform(
            self.gain[0], self.gain[1], 1)[0], inv=self.inv)).astype(inp.dtype)
        return (out-out.min())/(out.max()-out.min()+np.finfo(np.float32).eps)


def getContrastAugs(p=0.75):
    aug_dict = {
        AdjustSigmoid(): 0.30,
        AdjustLog(): 0.30,
        AdjustGamma(): 0.30,
        AdaptiveHistogramEqualization(): 0.10,
    }
    return torchvision.transforms.RandomApply([ApplyOneOf(aug_dict)], p=p)
    # TODO create params for everything

################# Contrast Augmentations #########################


class RandomCrop(SuperTransformer):
    def __init__(
            self,
            size: Union[Tuple[int], str],
            **kwargs
    ):
        super().__init__(**kwargs)
        if type(size) == str:
            size = tuple(int(tmp) for tmp in size.split(","))
        self.size = size

    def __crop2D(self, inp):
        h, w = inp.shape
        th, tw = self.size
        if w == tw and h == th:
            return inp
        i = np.random.randint(0, h - th + 1, size=(1, ))[0] if h-th > 0 else 0
        j = np.random.randint(0, w - tw + 1, size=(1, ))[0] if w-tw > 0 else 0
        return padIfNeeded(inp, self.size)[i:i+th, j:j+tw]

    def __crop3D(self, inp):
        w, h, d = inp.shape
        th, tw, td = self.size
        if w == tw and h == th and d == td:
            return inp
        i = np.random.randint(0, h - th + 1, size=(1, ))[0] if h-th > 0 else 0
        j = np.random.randint(0, w - tw + 1, size=(1, ))[0] if w-tw > 0 else 0
        k = np.random.randint(0, d - td + 1, size=(1, ))[0] if d-td > 0 else 0
        return padIfNeeded(inp, self.size)[i:i+th, j:j+tw, k:k+td]

    def apply(self, inp):
        return self.__crop2D(inp) if len(inp.shape) == 2 else self.__crop3D(inp)
