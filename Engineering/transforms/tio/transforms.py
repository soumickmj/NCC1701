from collections import defaultdict
from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform, FourierTransform, RandomMotion, RandomGhosting, Transform
from torchio import Motion, Ghosting
import torchio as tio
from skimage import exposure
from torchio import Subject

from Engineering.math.freq_trans import fftNc, ifftNc


class IntensityNorm(IntensityTransform):
    def __init__(
            self,
            type: str = "minmax",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                tensor = tensor.float()
                if self.type == "minmax":
                    transformed_tensor = (
                        tensor - tensor.min()) / (tensor.max() - tensor.min())
                elif self.type == "divbymax":
                    transformed_tensor = tensor / tensor.max()
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

class ForceAffine(Transform):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.inp.affine = subject.gt.affine
        return subject

class ChangeDataSpace(IntensityTransform):
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

    def apply_transform(self, subject: tio.Subject):
        if self.source_data_space == self.destin_data_space:
            return subject
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                if self.source_data_space == 0 and self.destin_data_space == 1:
                    transformed_tensor = fftNc(tensor, dim=self.data_dim)
                elif self.source_data_space == 1 and self.destin_data_space == 0:
                    transformed_tensor = ifftNc(tensor, dim=self.data_dim)
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

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

class RandomMotionGhosting(RandomTransform, IntensityTransform, FourierTransform):
    def __init__(
            self,
            degrees: Tuple[float] = (1.0, 3.0),
            translation: Tuple[float] = (1.0, 3.0),
            num_transforms: Tuple[int] = (2, 10),
            num_ghosts: Tuple[int] = (2, 5),
            intensity: Tuple[float] = (0.01, 0.75),
            restore: Tuple[float] = (0.01, 1.0),
            motion_image_interpolation: str = 'linear',
            ghosting_axes: Tuple[int] = (0, 1),
            p_motion: float = 1,
            p_ghosting: float = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.degrees = degrees
        self.translation = translation
        self.num_transforms = num_transforms
        self.num_ghosts = num_ghosts
        self.intensity = intensity
        self.restore = restore
        self.motion_image_interpolation = motion_image_interpolation
        self.ghosting_axes = ghosting_axes
        self.p_motion = p_motion
        self.p_ghosting = p_ghosting

    def apply_transform(self, subject):
        im = deepcopy(subject['gt'])
        degrees = np.random.uniform(
            low=self.degrees[0], high=self.degrees[1], size=1)
        translation = np.random.uniform(
            low=self.translation[0], high=self.translation[1], size=1)
        num_transforms = np.random.randint(
            low=self.num_transforms[0], high=self.num_transforms[1], size=(1))
        num_ghosts = np.random.randint(
            low=self.num_ghosts[0], high=self.num_ghosts[1], size=(1))
        intensity = np.random.uniform(
            low=self.intensity[0], high=self.intensity[1], size=1)
        restore = np.random.uniform(
            low=self.restore[0], high=self.restore[1], size=1)
        transform_rnd = tio.Compose([
            tio.transforms.RandomMotion(degrees=float(degrees[0]),
                                        translation=float(translation[0]),
                                        num_transforms=int(num_transforms[0]),
                                        image_interpolation=self.motion_image_interpolation,
                                        p=self.p_motion),
            tio.transforms.RandomGhosting(num_ghosts=int(num_ghosts[0]),
                                          axes=self.ghosting_axes,
                                          intensity=float(intensity[0]),
                                          restore=float(restore[0]),
                                          p=self.p_ghosting)
        ])
        subject.add_image(transform_rnd(im), "inp")
        return subject


class RandomMotionExtended(RandomMotion):
    def __init__(
        self,
        degrees: Tuple[float] = (1.0, 3.0),
        translation: Tuple[float] = (1.0, 3.0),
        num_transforms: Tuple[int] = (2, 10),
        image_interpolation: str = 'linear',
        **kwargs
    ):
        super().__init__(degrees=degrees, translation=translation,
                         num_transforms=2, image_interpolation=image_interpolation, **kwargs)  # num_transforms=2 is just dummy
        self.num_transforms = num_transforms

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        n_transforms = np.random.randint(
            low=self.num_transforms[0], high=self.num_transforms[1], size=(1))[0]
        for name, image in self.get_images_dict(subject).items():
            params = self.get_params(
                self.degrees_range,
                self.translation_range,
                n_transforms,
                is_2d=image.is_2d(),
            )
            times_params, degrees_params, translation_params = params
            arguments['times'][name] = times_params
            arguments['degrees'][name] = degrees_params
            arguments['translation'][name] = translation_params
            arguments['image_interpolation'][name] = self.image_interpolation
        transform = Motion(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed


class RandomGhostingExtended(RandomGhosting):
    def __init__(
        self,
        num_ghosts: Union[int, Tuple[int, int]] = (2, 5),
        axes: Union[int, Tuple[int]] = (0, 1),
        intensity: Union[float, Tuple[float, float]] = (0.01, 0.75),
        restore: float = (0.01, 1.0),
        **kwargs
    ):
        super().__init__(num_ghosts=num_ghosts, axes=axes, intensity=intensity,
                         restore=0.02, **kwargs)  # restore=0.02 is just dummy
        self.restore = restore

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        actual_restore = np.random.uniform(
            low=self.restore[0], high=self.restore[1], size=1)[0]
        if any(isinstance(n, str) for n in self.axes):
            subject.check_consistent_orientation()
        for name, image in self.get_images_dict(subject).items():
            is_2d = image.is_2d()
            axes = [a for a in self.axes if a != 2] if is_2d else self.axes
            params = self.get_params(
                self.num_ghosts_range,
                axes,
                self.intensity_range,
            )
            num_ghosts_param, axis_param, intensity_param = params
            arguments['num_ghosts'][name] = num_ghosts_param
            arguments['axis'][name] = axis_param
            arguments['intensity'][name] = intensity_param
            arguments['restore'][name] = actual_restore
        transform = Ghosting(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed


def getRandomMotionGhostingFast(
        degrees: Tuple[float] = (1.0, 3.0),
        translation: Tuple[float] = (1.0, 3.0),
        num_transforms: Tuple[int] = (2, 10),
        num_ghosts: Tuple[int] = (2, 5),
        intensity: Tuple[float] = (0.01, 0.75),
        restore: Tuple[float] = (0.01, 1.0),
        motion_image_interpolation: str = 'linear',
        ghosting_axes: Tuple[int] = (0, 1),
        p_motion: float = 1,
        p_ghosting: float = 1
):
    transform = tio.Compose([
        RandomMotionExtended(degrees=degrees,
                             translation=translation,
                             num_transforms=num_transforms,
                             image_interpolation=motion_image_interpolation,
                             p=p_motion),
        RandomGhostingExtended(num_ghosts=num_ghosts,
                               axes=ghosting_axes,
                               intensity=intensity,
                               restore=restore,
                               p=p_ghosting)
    ])
    return transform


class RandomMotionGhostingV2(RandomTransform, IntensityTransform, FourierTransform):
    def __init__(
            self,
            degrees: Tuple[float] = (1.0, 3.0),
            translation: Tuple[float] = (1.0, 3.0),
            num_transforms: Tuple[int] = (2, 10),
            num_ghosts: Tuple[int] = (2, 5),
            intensity: Tuple[float] = (0.01, 0.75),
            restore: Tuple[float] = (0.01, 1.0),
            image_interpolation: str = 'linear',
            ghosting_axes: Tuple[int] = (0, 1),
            p_motion: float = 1,
            p_ghosting: float = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        if type(degrees) == str:
            degrees = tuple([float(tmp) for tmp in degrees.split(",")])
        if type(translation) == str:
            translation = tuple([float(tmp) for tmp in translation.split(",")])
        if type(num_transforms) == str:
            num_transforms = tuple([int(tmp) for tmp in num_transforms.split(",")])
        if type(num_ghosts) == str:
            num_ghosts = tuple([int(tmp) for tmp in num_ghosts.split(",")])
        if type(intensity) == str:
            intensity = tuple([float(tmp) for tmp in intensity.split(",")])
        if type(restore) == str:
            restore = tuple([float(tmp) for tmp in restore.split(",")])
        if type(ghosting_axes) == str:
            ghosting_axes = tuple([int(tmp) for tmp in ghosting_axes.split(",")])

        self.transform = getRandomMotionGhostingFast(
            degrees=degrees,
            translation=translation,
            num_transforms=num_transforms,
            num_ghosts=num_ghosts,
            intensity=intensity,
            restore=restore,
            motion_image_interpolation=image_interpolation,
            ghosting_axes=ghosting_axes,
            p_motion=p_motion,
            p_ghosting=p_ghosting
        )

    def apply_transform(self, subject):
        im = deepcopy(subject['gt'])
        subject.add_image(self.transform(im), "inp")
        return subject
