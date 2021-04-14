#!/usr/bin/env python

"""
This module contains all the Transforms that can be applied on the MRI PyTorch Custom Dataset
All the transforms are written as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it’s called.
Contains a couple of dummy transforms code
"""


import numpy as np
import torch
from utils.HandleNifti import Nifti3Dto2D, Nifti2Dto3D

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Many of the Transforms finished. But more transforms needs to be added. Contains some dummy code for some transforms"

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Achtung! - This is a dummy code taken from PyTorch Doc. Not yet implimented for our requirement (MRITorchDS)

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Achtung! - This is a dummy code taken from PyTorch Doc. Not yet implimented for our requirement (MRITorchDS)

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class RemoveVerySmallEValues(object):
    """Normalizes the input 2D or 3D image within 0 to 1"""

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, sample):
        fully, under = np.nan_to_num(sample['fully']), np.nan_to_num(sample['under'])   
        fully[fully < self.delta] = 0
        under[under < self.delta] = 0
        return {'fully': fully,
                'under': under,
                'subjectName': sample['subjectName'],
                'fileName': sample['fileName']}

class MinMaxNormalization(object):
    """Normalizes the input 2D or 3D image within 0 to 1"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under']        
        import warnings
        warnings.filterwarnings('error')
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
        return {'fully': fullyNorm,
                'under': underNorm,
                'subjectName': sample['subjectName'],
                'fileName': sample['fileName']}

class Convert3Dto2D(object):
    """Convert 3D Vols to 2D"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under']

        fully = Nifti3Dto2D(fully)
        under = Nifti3Dto2D(under)
        return {'fully': fully,
                'under': under,
                'subjectName': sample['subjectName'],
                'fileName': sample['fileName']}

class ConvertToSuitableType(object):
    """Convert 3D Vols to 2DMultiSlice or 2DSingleSlice"""

    def To2DSingleSlice(self, sample):
        fully, under = sample['fully'], sample['under']
        sample['fully'] = sample['fully'][:,self.sliceno,:,:]
        sample['under'] = sample['under'][:,self.sliceno,:,:]    
        return sample

    def ToSelectSlice(self, sample):
        fully, under = sample['fully'], sample['under']
        sample['fully'] = sample['fully'][:,self.startingSelectSlice:self.endingSelectSlice,:,:]
        sample['under'] = sample['under'][:,self.startingSelectSlice:self.endingSelectSlice,:,:]    
        return sample

    def __init__(self, type, sliceno=None, startingSelectSlice=None, endingSelectSlice=None):
        self.sliceno = sliceno
        self.startingSelectSlice = startingSelectSlice
        self.endingSelectSlice = endingSelectSlice
        if(type == '2DSingleSlice'):
            self.type_transform = lambda x: self.To2DSingleSlice(x)
        elif(type == '2DSelectMultiSlice' or type == '3DSelectSlice'):
            self.type_transform = lambda x: self.ToSelectSlice(x)
        else:
            self.type_transform = lambda x: x

    def __call__(self, sample):
        return self.type_transform(sample)

class ToTensor2D(object):
    """Convert ndarrays in sample to Tensors.
       This for 2D Images. To be used only with 2D Images those are already converted to 2D from 3D
       Specialized Transform for 3D Vols is created seperately"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under']

        # swap channel axis because
        # numpy image/MR Image: H x W x C
        # torch image: C X H X W
        # If channel info not present, then add a fake axis for channel
        if len(np.shape(fully)) == 3:
            fully = fully.transpose((2, 0, 1))
        else:
            fully = np.expand_dims(fully, 0)
        if len(np.shape(under)) == 3:
            under = under.transpose((2, 0, 1))
        else:
            under = np.expand_dims(under, 0)
        return {'fully': torch.from_numpy(fully).float(),
                'under': torch.from_numpy(under).float(),
                'subjectName': sample['subjectName'],
                'fileName': sample['fileName']}

class ToTensor3D(object):
    """Convert ndarrays in sample to Tensors.
       Specilized Transform for 3D Vols"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under']

        # swap channel axis because
        # numpy image/MR Image: H x W x D X C / X x Y x Z x C
        # torch image: C X H X W X D / C x X x Y x Z #From original info, but now I see it different in torch website, so changed it
        # New torch image : C X D X H X W / C x Z x X x Y
        fully = fully.transpose((3, 2, 0, 1)) #originally: fully = fully.transpose((3, 0, 1, 2))
        under = under.transpose((3, 2, 0, 1)) #originally: under = under.transpose((3, 0, 1, 2))
        if(np.iscomplex(fully).any()):
            fully = np.concatenate((fully.real, fully.imag))
            under = np.concatenate((under.real, under.imag))
        return {'fully': torch.from_numpy(fully).float(),
                'under': torch.from_numpy(under).float(),
                'subjectName': sample['subjectName'],
                'fileName': sample['fileName']}

class FromTensorToNumpy2D(object):
    """Convert Tensors in ouput to ndarrays.
       This for 2D Images. To be used only when the network provides output in terms of 2D Images"""

    def __call__(self, tensor):
        array = tensor.numpy()
        array = array.transpose((1, 2, 0))

        return array

class FromTensorToNumpy2DToNumpy3D(object):
    """Convert Tensors in ouput to ndarrays.
       This for 2D Images. To be used only when the network provides output in terms of 2D Images"""

    def __call__(self, tensor):
        array = tensor.numpy()
        array = array.transpose((1, 2, 0))

        array = Nifti2Dto3D(array)

        return array

class FromTensorToNumpy3D(object):
    """Convert Tensors in ouput to ndarrays.
       This for 2D Images. To be used only when the network provides output in terms of 2D Images"""

    def __call__(self, tensor):
        array = tensor.numpy()
        if(np.shape(array)[0] == 2):
            array = array[0,:,:,:] + 1j * array[1,:,:,:]
            array = np.expand_dims(array, 0)
        #array = np.apply_along_axis(lambda args: [complex(*args)], 0, array)
        array = array.transpose((2, 3, 1, 0)) #originally: array = array.transpose((1, 2, 3, 0))

        return array

class Rescale4Unet(object):
    """Rescale the image in a sample to a given size.

    Achtung! - This is a dummy code taken from PyTorch Doc. Not yet implimented for our requirement (MRITorchDS)

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}