import torch
from Engineering.math.freq_trans import fftNc, ifftNc
from typing import Union, Tuple, Callable, Optional, Sequence, Dict
from copy import deepcopy
import numpy as np

##################Master Classes###########################

class SuperTransformer():
    def __init__(
            self,
            p: float = 1,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            applyonly: bool = False, #To skip all sample-level processes and all params, just simply call apply on the supplied tensor
            gt2inp: bool = False, 
            **kwargs
            ):
        self.p = p
        self.include = [include] if type(include) is str else include
        self.exclude = [exclude] if type(exclude) is str else exclude
        self.applyonly = applyonly
        self.gt2inp = gt2inp

    def __call__(self, sample):
        if self.applyonly:
            return self.apply(sample)
        if self.gt2inp:
            if torch.rand(1).item() > self.p:
                sample['inp'] = deepcopy(sample['gt'])
            else:
                sample['inp'] = {
                    'data': self.apply(sample['gt']['data']),
                    'path': ""
                }
        else:
            if torch.rand(1).item() > self.p:
                return sample
            for k in sample.keys():
                if (type(sample[k]) is not dict) or ("data" not in sample[k]) or (bool(self.include) and k not in self.include) or (not bool(self.include) and bool(self.exclude) and k in self.exclude):
                    continue
                sample[k]['data'] = self.apply(sample[k]['data'])
        return sample

class ApplyOneOf():
    def __init__(
            self,
            transforms_dict
            ):
        self.transforms_dict = transforms_dict

    def __call__(self, inp):
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        return transform(inp)

###########################################################

################Transformation Functions###################

def padIfNeeded(inp, size=None):
    inp_shape = inp.shape
    pad = [(0,0),]*len(inp_shape)
    pad_requried = False
    for i in range(len(inp_shape)):
        if inp_shape[i] < size[i]:
            diff = size[i]-inp_shape[i]
            pad[i] = (diff//2, diff-(diff//2))
            pad_requried = True
    if not pad_requried:
        return inp
    else:
        return np.pad(inp, pad)

def cropcentreIfNeeded(inp, size=None):
    if len(inp.shape)==2:
        w, h = inp.shape
    else:
        w, h, d = inp.shape
    if bool(size[0]) and h > size[0]:
        diff = h-size[0]
        inp = inp[diff//2:diff//2+size[0],...]
    if bool(size[1]) and w > size[1]:
        diff = w-size[1]
        if len(inp.shape)==2:
            inp = inp[..., diff//2:diff//2+size[1]]
        else:
            inp = inp[:, diff//2:diff//2+size[1], :]
    if len(inp.shape) == 3 and d > size[2]:
        diff = d-size[2]
        inp = inp[..., diff//2:diff//2+size[2]]
    return inp

class CropOrPad(SuperTransformer):
    def __init__(
            self,
            size: Union[Tuple[int], str],
            **kwargs
    ):
        super().__init__(**kwargs)
        if type(size) == str:
            size = tuple([int(tmp) for tmp in size.split(",")])
        self.size = size

    def apply(self, inp):
        inp = padIfNeeded(inp, size=self.size)
        return cropcentreIfNeeded(inp, size=self.size)

class IntensityNorm(SuperTransformer):
    def __init__(
            self,
            type: str = "minmax",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type

    def apply(self, inp):
        if self.type == "minmax":
            return (inp - inp.min()) / (inp.max() - inp.min() + np.finfo(np.float32).eps)
        elif self.type == "divbymax":
            return inp / (inp.max() + np.finfo(np.float32).eps)

class IntensityNorm(SuperTransformer):
    def __init__(
            self,
            type: str = "minmax",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type

    def apply(self, inp):
        if self.type == "minmax":
            return (inp - inp.min()) / (inp.max() - inp.min() + np.finfo(np.float32).eps)
        elif self.type == "divbymax":
            return inp / (inp.max() + np.finfo(np.float32).eps)

class CutNoise(SuperTransformer):
    def __init__(
            self,
            level: float = 0.07,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.level = level

    def apply(self, inp):
        inp[inp <= self.level] = 0
        return (inp-inp.min())/(inp.max()-inp.min()+np.finfo(np.float32).eps)

class ChangeDataSpace(SuperTransformer):
    def __init__(
            self,
            source_data_space, 
            destin_data_space,
            data_dim = (-3,-2,-1),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.source_data_space = source_data_space
        self.destin_data_space = destin_data_space
        self.data_dim = data_dim

    def apply(self, inp):
        if self.source_data_space == 0 and self.destin_data_space == 1:
            return fftNc(inp, dim=self.data_dim)
        elif self.source_data_space == 1 and self.destin_data_space == 0:
            return ifftNc(inp, dim=self.data_dim)

def getDataSpaceTransforms(dataspace_inp, model_dataspace_inp, dataspace_gt, model_dataspace_gt):
    if dataspace_inp == dataspace_gt and model_dataspace_inp == model_dataspace_gt and dataspace_inp != model_dataspace_inp:
        return [ChangeDataSpace(dataspace_inp, model_dataspace_inp)]
    else:
        trans = []
        if dataspace_inp != model_dataspace_inp and dataspace_inp != -1 and model_dataspace_inp != -1:
            trans.append(ChangeDataSpace(dataspace_inp, model_dataspace_inp, include="inp"))
        elif dataspace_gt != model_dataspace_gt and dataspace_gt != -1 and model_dataspace_gt != -1:
            trans.append(ChangeDataSpace(dataspace_gt, model_dataspace_gt, include="gt"))
        return trans

###########################################################