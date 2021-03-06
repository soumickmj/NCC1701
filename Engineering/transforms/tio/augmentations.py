from typing import Tuple

import torch
import torchio as tio
import torchvision
from Engineering.transforms import augmentations as cusAugs
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
        self.transformer = cusAugs.AdaptiveHistogramEqualization(
            kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins, applyonly=True)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = self.transformer(tensor)
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
        self.transformer = cusAugs.AdjustGamma(
            gamma=gamma, gain=gain, applyonly=True)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = self.transformer(tensor)
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
        self.transformer = cusAugs.AdjustSigmoid(
            cutoff=cutoff, gain=gain, inv=inv, applyonly=True)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = self.transformer(tensor)
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
        self.transformer = cusAugs.AdjustLog(
            gain=gain, inv=inv, applyonly=True)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                isPYTTensor = type(tensor) is torch.Tensor
                tensor = tensor.cpu().numpy() if isPYTTensor else tensor
                transformed_tensor = self.transformer(tensor)
                transformed_tensors.append(torch.from_numpy(
                    transformed_tensor) if isPYTTensor else transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


def getContrastAugs(p=0.75):
    aug_dict = {
        AdjustSigmoid(): 0.30,
        AdjustLog(): 0.30,
        AdjustGamma(): 0.30,
        AdaptiveHistogramEqualization(): 0.10,
    }
    return torchvision.transforms.RandomApply([tio.OneOf(aug_dict)], p=p)
    # TODO create params for everything
