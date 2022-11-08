from collections import defaultdict
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import torch
import torchio as tio
from torchio import Ghosting, Motion, Subject
from torchio.transforms import (FourierTransform, IntensityTransform,
                                RandomGhosting, RandomMotion, Transform)
from torchio.transforms.augmentation import RandomTransform

########### RandomMotionGhosting ##############

### Basic Version ###


class RandomMotionGhosting(RandomTransform, IntensityTransform, FourierTransform):
    """
        Combines RandomMotion and RandomGhosting transforms of TorchIO into one transformation. 
        Randomly composes them during the application process for each subject
    """

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

### Support class(es) and function(s) for the Advanced version ###


class RandomMotionExtended(RandomMotion):
    """
        Extends RandomMotion and gives the abbility to select the level randomly for each subject within a given range
    """

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
        return transform(subject)


class RandomGhostingExtended(RandomGhosting):
    """
        Extends RandomGhosting and gives the abbility to select the level randomly for each subject within a given range
    """

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
        return transform(subject)


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
    """
        Function that combines RandomMotion and RandomGhosting transforms of TorchIO into one transformation and returns a composed transformation
    """
    return tio.Compose(
        [
            RandomMotionExtended(
                degrees=degrees,
                translation=translation,
                num_transforms=num_transforms,
                image_interpolation=motion_image_interpolation,
                p=p_motion,
            ),
            RandomGhostingExtended(
                num_ghosts=num_ghosts,
                axes=ghosting_axes,
                intensity=intensity,
                restore=restore,
                p=p_ghosting,
            ),
        ]
    )

### Extended / Faster Version ###


class RandomMotionGhostingFast(RandomTransform, IntensityTransform, FourierTransform):
    """
        Combines RandomMotion and RandomGhosting transforms of TorchIO into one transformation. 
        Instead of Randomly composing them during the application process for each subject, this version uses the "Extened" version of those transforms to be able to
        apply without re-creating objects for each
    """

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
            degrees = tuple(float(tmp) for tmp in degrees.split(","))
        if type(translation) == str:
            translation = tuple(float(tmp) for tmp in translation.split(","))
        if type(num_transforms) == str:
            num_transforms = tuple(int(tmp) for tmp in num_transforms.split(","))
        if type(num_ghosts) == str:
            num_ghosts = tuple(int(tmp) for tmp in num_ghosts.split(","))
        if type(intensity) == str:
            intensity = tuple(float(tmp) for tmp in intensity.split(","))
        if type(restore) == str:
            restore = tuple(float(tmp) for tmp in restore.split(","))
        if type(ghosting_axes) == str:
            ghosting_axes = tuple(int(tmp) for tmp in ghosting_axes.split(","))

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

########### RandomMotionGhosting ##############
