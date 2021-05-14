# MinMaxNormalization
# ToTensor3D
# ConvertToSuitableType

#!/usr/bin/env python

"""
This module contains all the Transforms that can be applied on the MRI PyTorch Custom Dataset
All the transforms are written as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it’s called.
Contains a couple of dummy transforms code

Includes transforms taken from fastMRI, where the copyright belongs to Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
# import kornia.geometry.transform as kortrans
# from scipy.misc import imrotate
from skimage.transform import rotate
import random
# from utils.HandleNifti import Nifti3Dto2D, Nifti2Dto3D
# from Math.FrequencyTransforms import fft2c, ifft2c
# from utils.fastMRI.TorchDSTransforms import fft2 as t_fft2, ifft2 as t_ifft2, complex_abs as t_complex_abs, roll as t_roll

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Many of the Transforms finished. But more transforms needs to be added. Contains some dummy code for some transforms"

class MinMaxNormalization(object):
    """Normalizes the input 2D or 3D image within 0 to 1"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under'] 
        try:            
            fullyNorm = ((fully - fully.min()) / (fully.max() - fully.min()))
        except RuntimeWarning as err:
            print("Runtime error in MixMax of Fully: {0}".format(err))
            fullyNorm = fully
        try:            
            underNorm = ((under - under.min()) / (under.max() - under.min()))          
        except RuntimeWarning as err:
            print("Runtime error in MixMax of Under: {0}".format(err))
            underNorm = under
        sample['fully'] = fullyNorm
        sample['under'] = underNorm
        return sample