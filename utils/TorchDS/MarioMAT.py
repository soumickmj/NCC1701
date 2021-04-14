#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of OASIS1

"""

import os
import sys
from torch.utils.data import Dataset, DataLoader
from utils.GetROI import ClearNonROI
from Math.FrequencyTransforms import fft2c, rfft2c, fht2c

import scipy.io as sio

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class MRITorchDS(Dataset):
    """Read MR Images (Under and Fully parallely) as Torch (custom) Dataset"""

    def __init__(self, mat_path, domain='abs_fourier', transform=None, getROIMode=None):
        """
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to the OASIS1 standard
        Args:
            root_fully (string): Directory with all the images - fully sampled.
            root_under (string): Directory with all the images - under sampled.
            fileExtension_under (string): 
            domain (string): In which domain network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image | 
            transform (callable, optional): Optional transform to be applied on each sample. It can be custom composed transform, or single pre-defined transform
        Both 'root_fully' and 'root_under' needs to follow above mentioned folder sturcture
        Files should be organized identically in both 'root_fully' and 'root_under'
        """        
        self.mat_path = mat_path
        self.transform = transform
        self.mat = sio.loadmat(self.mat_path)
        
        if(domain == 'fourier'):
            self.domain_transform = lambda x: fft2c(x) #Matlab iquivalent: domain_transform = @fft2c
        elif(domain == 'real_fourier'):
            self.domain_transform = lambda x: rfft2c(x)
        elif(domain == 'abs_of_fourier'):
            self.domain_transform = lambda x: abs(fft2c(x))
        elif(domain == 'real_of_fourier'):
            self.domain_transform = lambda x: fft2c(x).real
        elif(domain == 'abs_of_real_fourier'):
            self.domain_transform = lambda x: abs(rfft2c(x))
        elif(domain == 'real_of_real_fourier'):
            self.domain_transform = lambda x: rfft2c(x).real
        elif(domain == 'hartley'):
            self.domain_transform = lambda x: fht2c(x)
        elif(domain == 'image'):
            self.domain_transform = lambda x: x
        else:
            print('Invalid Domain Specified')
            sys.exit(0)
        #if(getROIMode == 'FromUnderToFully'):
        #    self.getROI = lambda fully, under: (self.roiFromUnderToFully(fully, under), under) 
        #elif (getROIMode == 'FromFullyToUnder'):
        #    self.getROI = lambda fully, under: (fully, self.roiFromFullyToUnder(fully, under))
        #else:
        #    self.getROI = lambda fully, under: (fully, under)

    def roiFromUnderToFully(self, fully, under):
        return ClearNonROI(under, fully)

    def roiFromFullyToUnder(self, fully, under):
        return ClearNonROI(fully, under)

    def __len__(self):
        """For returning the length of the file list"""
        return len(self.mat['fully'])

    def __getitem__(self, idx):
        """For returning one sample from file_list
        at index idx
        after applying transform on them"""
        #(vol_fully, vol_under) = self.getROI(FileRead3D(self.file_list[idx]['path2fully']), FileRead3D(self.file_list[idx]['path2under']))

        vol_fully = self.domain_transform(self.mat['fully'][idx,:,:])
        vol_under = self.domain_transform(self.mat['under'][idx,:,:])
        #vol_artifact = self.domain_transform(self.mat['artifact'][idx,:,:])

        sample = {
            'fully': vol_fully,
            'under': vol_under,
            #'artifact': vol_artifact,
            'subjectName': 'unknown',
            'fileName': idx
            }

        if self.transform: #If transorm been supplied, apply it. It can be composed transform inlcuding custom transformations
            sample = self.transform(sample) 

        return sample

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)
        