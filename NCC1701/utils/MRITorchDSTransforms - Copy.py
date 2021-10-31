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
import kornia.geometry.transform as kortrans
# from scipy.misc import imrotate
from skimage.transform import rotate
import random
from utils.HandleNifti import Nifti3Dto2D, Nifti2Dto3D
from Math.FrequencyTransforms import fft2c, ifft2c
from utils.fastMRI.TorchDSTransforms import fft2 as t_fft2, ifft2 as t_ifft2, complex_abs as t_complex_abs, roll as t_roll

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Many of the Transforms finished. But more transforms needs to be added. Contains some dummy code for some transforms"

class RemoveVerySmallEValues(object):
    """Normalizes the input 2D or 3D image within 0 to 1"""

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, sample):
        fully, under = np.nan_to_num(sample['fully']), np.nan_to_num(sample['under'])   
        fully[fully < self.delta] = 0
        under[under < self.delta] = 0
        sample['fully'] = fully
        sample['under'] = under
        return sample

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
        sample['fully'] = fullyNorm
        sample['under'] = underNorm
        return sample

class Convert3Dto2D(object):
    """Convert 3D Vols to 2D"""

    def __call__(self, sample):
        fully, under = sample['fully'], sample['under']

        fully = Nifti3Dto2D(fully)
        under = Nifti3Dto2D(under)
        sample['fully'] = fully
        sample['under'] = under
        return sample

class ConvertToSuitableType(object):
    """Convert 3D Vols to 2DMultiSlice or 2DSingleSlice"""

    def To2DSingleSlice(self, sample):
        fully, under = sample['fully'], sample['under']
        sample['fully'] = sample['fully'][:,self.sliceno,...]
        sample['under'] = sample['under'][:,self.sliceno,...]    
        return sample

    def ToSelectSlice(self, sample):
        fully, under = sample['fully'], sample['under']
        sample['fully'] = sample['fully'][:,self.startingSelectSlice:self.endingSelectSlice,:,:]
        sample['under'] = sample['under'][:,self.startingSelectSlice:self.endingSelectSlice,:,:]    
        return sample

    def __init__(self, type, sliceno=None, startingSelectSlice=None, endingSelectSlice=None):
        self.sliceno = sliceno
        if self.sliceno is None:
            self.sliceno = 0
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

        sample['fully'] = torch.from_numpy(fully).float()
        sample['under'] = torch.from_numpy(under).float()
        return sample

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

        sample['fully'] = torch.from_numpy(fully).float()
        sample['under'] = torch.from_numpy(under).float()
        return sample

class TransformFullyUpdateUnder(object):
    """To reuse the already created fully and under numpy arrays, apply new transforms on top of fully, supplied as transform and generate new data and store it (overwrite) in under"""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, sample):
        if self.transform is not None:
            sample['under'] = self.transform(sample['fully'].copy())
        return sample


class TransformAndUpdateFullyUnder(object):
    """To reuse the already created fully and under numpy arrays, apply new transforms on top of them supplied as transform and generate new data and store it (overwrite)"""

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, sample):
        if self.transform is not None:
            sample['fully'] = self.transform(sample['fully'])
            sample['under'] = self.transform(sample['under'])
        return sample

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


class CorruptByMotionNP(object):
    """To introduce motion artefacts.
    Send numpy arrays in the call function"""

    def __init__(self, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False):
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        self.rotation_limit = rotation_limit
        self.trans_x_limit = trans_x_limit
        self.trans_y_limit = trans_y_limit
        self.corruption_dir = corruption_dir
        self.normalize_rotation = normalize_rotation

    def performCorruption(self, img):
        #Introduce Motion
        if self.corruption_dir is None:
            tmp_pe_ = random.randint(0, 1)
        else:
            tmp_pe_ = self.corruption_dir
        aux = np.zeros(img.shape, dtype=np.complex)
        rotation = 5#random.uniform(0.1, self.rotation_limit)
        for i in range(0, img.shape[tmp_pe_]):
            tmp_rot = rotation * random.randint(-1, 1)
            # img_rot = imrotate(img, tmp_rot, interp='bicubic')
            img_rot = rotate(img, tmp_rot, order=3)
            ksp_rot = fft2c(img_rot)
            if tmp_pe_ == 0:
                aux[i, :] = ksp_rot[i, :]
            else:
                aux[:, i] = ksp_rot[:, i]
        img = np.abs(ifft2c(aux))

        #Introduce translation
        trans_x = random.randint(-self.trans_x_limit, self.trans_x_limit)
        trans_y = random.randint(-self.trans_y_limit, self.trans_y_limit)
        img = np.roll(img, trans_x, axis=0)
        img = np.roll(img, trans_y, axis=1)

        corruptions = {}
        if self.normalize_rotation:
            corruptions['rotation'] = rotation / self.rotation_limit
        else:
            corruptions['rotation'] = rotation
        corruptions['rotation_limit'] = self.rotation_limit
        corruptions['rotation_normalized'] = self.normalize_rotation
        corruptions['corruption_dir'] = tmp_pe_
        corruptions['trans_x'] = trans_x
        corruptions['trans_y'] = trans_y

        return img, corruptions

    def __call__(self, img):
        if len(img.shape) == 3:
            corruptions = []
            for i in range(img.shape[0]):
                img[i,...], temp = self.performCorruption(img[i,...])
                corruptions.append(temp)
        elif len(img.shape) == 2:
            img, corruptions = self.performCorruption(img)
        else:
            import sys
            sys.exit()

        return img, corruptions


class CorruptByMotionNPMultiCoil(object):
    """To introduce motion artefacts.
    Send numpy arrays in the call function"""

    def __init__(self, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False):
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        self.rotation_limit = rotation_limit
        self.trans_x_limit = trans_x_limit
        self.trans_y_limit = trans_y_limit
        self.corruption_dir = corruption_dir
        self.normalize_rotation = normalize_rotation

    def performCorruption(self, img):
        #Introduce Motion
        if self.corruption_dir is None:
            tmp_pe_ = random.randint(0, 1)
        else:
            tmp_pe_ = self.corruption_dir
        aux = np.zeros(img.shape, dtype=np.complex)
        rotation = random.uniform(0.1, self.rotation_limit)
        for i in range(0, img.shape[tmp_pe_+1]):
            tmp_rot = rotation * random.randint(-1, 1)
            for j in range(0, img.shape[0]):
                # img_rot = imrotate(img[j], tmp_rot, interp='bicubic')
                img_rot = rotate(img[j], tmp_rot, order=3)
                ksp_rot = fft2c(img_rot)
                if tmp_pe_ == 0:
                    aux[j, i, :] = ksp_rot[i, :]
                else:
                    aux[j, :, i] = ksp_rot[:, i]
        img = np.transpose(np.abs(ifft2c(np.transpose(aux, (1,2,0)))), (2,0,1))

        #Introduce translation
        trans_x = random.randint(-self.trans_x_limit, self.trans_x_limit)
        trans_y = random.randint(-self.trans_y_limit, self.trans_y_limit)
        img = np.roll(img, trans_x, axis=0)
        img = np.roll(img, trans_y, axis=1)

        corruptions = {}
        if self.normalize_rotation:
            corruptions['rotation'] = rotation / self.rotation_limit
        else:
            corruptions['rotation'] = rotation
        corruptions['rotation_limit'] = self.rotation_limit
        corruptions['rotation_normalized'] = self.normalize_rotation
        corruptions['corruption_dir'] = tmp_pe_
        corruptions['trans_x'] = trans_x
        corruptions['trans_y'] = trans_y

        return img, corruptions

    def __call__(self, img):
        if len(img.shape) == 4:
            corruptions = []
            for i in range(img.shape[0]):
                img[i,...], temp = self.performCorruption(img[i,...])
                corruptions.append(temp)
        elif len(img.shape) == 3:
            img, corruptions = self.performCorruption(img)
        else:
            import sys
            sys.exit()

        return img, corruptions

class CorruptByMotionNPMultiCoilTorch(object):
    """To introduce motion artefacts.
    Send numpy arrays in the call function"""

    def __init__(self, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False):
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        self.rotation_limit = rotation_limit
        self.trans_x_limit = trans_x_limit
        self.trans_y_limit = trans_y_limit
        self.corruption_dir = corruption_dir
        self.normalize_rotation = normalize_rotation

    def performCorruption(self, img):
        #Introduce Motion
        if self.corruption_dir is None:
            tmp_pe_ = random.randint(0, 1)
        else:
            tmp_pe_ = self.corruption_dir
        aux = np.zeros(img.shape, dtype=np.complex)
        rotation = random.uniform(0.1, self.rotation_limit)
        for i in range(0, img.shape[tmp_pe_+1]):
            tmp_rot = rotation * random.randint(-1, 1)
            for j in range(0, img.shape[0]):
                # img_rot = imrotate(img[j], tmp_rot, interp='bicubic')
                img_rot = rotate(img[j], tmp_rot, order=3)
                ksp_rot = fft2c(img_rot)
                if tmp_pe_ == 0:
                    aux[j, i, :] = ksp_rot[i, :]
                else:
                    aux[j, :, i] = ksp_rot[:, i]
        img = np.transpose(np.abs(ifft2c(np.transpose(aux, (1,2,0)))), (2,0,1))

        #Introduce translation
        trans_x = random.randint(-self.trans_x_limit, self.trans_x_limit)
        trans_y = random.randint(-self.trans_y_limit, self.trans_y_limit)
        img = np.roll(img, trans_x, axis=0)
        img = np.roll(img, trans_y, axis=1)

        corruptions = {}
        if self.normalize_rotation:
            corruptions['rotation'] = rotation / self.rotation_limit
        else:
            corruptions['rotation'] = rotation
        corruptions['rotation_limit'] = self.rotation_limit
        corruptions['rotation_normalized'] = self.normalize_rotation
        corruptions['corruption_dir'] = tmp_pe_
        corruptions['trans_x'] = trans_x
        corruptions['trans_y'] = trans_y

        return img, corruptions

    def __call__(self, img):
        if len(img.shape) == 4:
            corruptions = []
            for i in range(img.shape[0]):
                img[i,...], temp = self.performCorruption(img[i,...])
                corruptions.append(temp)
        elif len(img.shape) == 3:
            img, corruptions = self.performCorruption(img)
        else:
            import sys
            sys.exit()

        return img, corruptions

class CorruptByMotionTorch(object):
    """To introduce motion artefacts.
    Send numpy arrays in the call function"""

    def __init__(self, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False):
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        self.rotation_limit = rotation_limit
        self.trans_x_limit = trans_x_limit
        self.trans_y_limit = trans_y_limit
        self.corruption_dir = corruption_dir
        self.normalize_rotation = normalize_rotation

    def performCorruption(self, img):
        #Introduce Motion
        if self.corruption_dir is None:
            tmp_pe_ = random.randint(0, 1)
        else:
            tmp_pe_ = self.corruption_dir
        aux = torch.zeros(img.shape+(2,), dtype=torch.float).to(img.device)
        rotation = random.uniform(0.1, self.rotation_limit)
        for i in range(0, img.shape[tmp_pe_]):
            tmp_rot = rotation * random.randint(-1, 1)
            angle = torch.ones(1).to(img.device) * tmp_rot

            img_rot = kortrans.rotate(img.unsqueeze(0).unsqueeze(0).float(), angle).squeeze()
            ksp_rot = t_fft2(img_rot)
            if tmp_pe_ == 0:
                aux[i, :, :] = ksp_rot[i, :, :]
            else:
                aux[:, i, :] = ksp_rot[:, i, :]
        img = t_complex_abs(t_ifft2(aux))

        #Introduce translation
        trans_x = random.randint(-self.trans_x_limit, self.trans_x_limit)
        trans_y = random.randint(-self.trans_y_limit, self.trans_y_limit)
        img = t_roll(img, trans_x, 0)
        img = t_roll(img, trans_y, 1)

        corruptions = {}
        if self.normalize_rotation:
            corruptions['rotation'] = rotation / self.rotation_limit
        else:
            corruptions['rotation'] = rotation
        corruptions['rotation_limit'] = self.rotation_limit
        corruptions['rotation_normalized'] = self.normalize_rotation
        corruptions['corruption_dir'] = tmp_pe_
        corruptions['trans_x'] = trans_x
        corruptions['trans_y'] = trans_y

        return img, corruptions

    def __call__(self, img):
        if len(img.shape) == 3:
            corruptions = []
            for i in range(img.shape[0]):
                img[i,...], temp = self.performCorruption(img[i,...])
                corruptions.append(temp)
        elif len(img.shape) == 2:
            img, corruptions = self.performCorruption(img)
        else:
            import sys
            sys.exit()

        return img, corruptions

class NormalizeTensor(object):
    """To introduce motion artefacts.
    Send numpy arrays in the call function"""

    def __init__(self, type='minmax'):
        #If rotation is none, then a 
        self.norm_type = type

    def __call__(self, tensor):
        if self.norm_type == 'minmax':
            try:
                return ((tensor - tensor.min()) / (tensor.max() - tensor.min()))
            except:
                return tensor / tensor.max()
        elif self.norm_type == 'divmax':
            return tensor / tensor.max()
        else:
            import sys
            sys.exit()

        return img