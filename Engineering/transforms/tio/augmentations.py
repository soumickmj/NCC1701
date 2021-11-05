from typing import Tuple

import numpy as np
import torch
import torchio as tio
from skimage import exposure
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform


class AdaptiveHistogramEqualization(RandomTransform, IntensityTransform):
    def __init__(
            self,
            kernel_size: Tuple[int] = (10, 100),
            clip_limit: float = 0.01,
            nbins: int = 512,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = exposure.equalize_adapthist(tensor, kernel_size=np.random.randint(
                    self.kernel_size[0], high=self.kernel_size[1], size=(1))[0], clip_limit=self.clip_limit, nbins=self.nbins).astype(tensor.dtype)
                transformed_tensors.append(torch.from_numpy(
                    transformed_tensor) if isPYTTensor else transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


class AdjustGamma(RandomTransform, IntensityTransform):
    def __init__(
            self,
            gamma: Tuple[float] = (0.5, 3.0),
            gain: float = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.gain = gain

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = exposure.adjust_gamma(tensor, gamma=np.random.uniform(
                    self.gamma[0], self.gamma[1], 1)[0], gain=self.gain).astype(tensor.dtype)
                transformed_tensors.append(torch.from_numpy(
                    transformed_tensor) if isPYTTensor else transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


class AdjustSigmoid(RandomTransform, IntensityTransform):
    def __init__(
            self,
            cutoff: Tuple[float] = (0.01, 0.75),
            gain: Tuple[int] = (2, 10),
            inv: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.gain = gain
        self.inv = inv

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = exposure.adjust_sigmoid(tensor, cutoff=np.random.uniform(self.cutoff[0], self.cutoff[1], 1)[0],
                                                             gain=np.random.randint(
                                                                 self.gain[0], high=self.gain[1], size=(1))[0],
                                                             inv=self.inv).astype(tensor.dtype)
                transformed_tensors.append(torch.from_numpy(
                    transformed_tensor) if isPYTTensor else transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


class AdjustLog(RandomTransform, IntensityTransform):
    def __init__(
            self,
            gain: Tuple[float] = (-0.5, 0.5),
            inv: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.inv = inv

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = np.abs(exposure.adjust_log(tensor, gain=np.random.uniform(
                    self.gain[0], self.gain[1], 1)[0], inv=self.inv)).astype(tensor.dtype)
                transformed_tensors.append(torch.from_numpy(
                    transformed_tensor) if isPYTTensor else transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

def getContrastAugs():
    aug_dict = {
            AdjustSigmoid(): 0.30,
            AdjustLog(): 0.30,
            AdjustGamma(): 0.30,
            AdaptiveHistogramEqualization(): 0.10,
        }
    return tio.OneOf(aug_dict)
    #TODO create params for everything