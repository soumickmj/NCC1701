#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset from MAT files, previously generated from any custom torch DS using TorchDS2MAT
It is assumed that the MAT files would contain 4 keys - fully, under, fileName and subjectName
fully and under has to be converted to torch Tensors

MAT files should have only one subject each (batch size 1 while creating the MATs).
But while using this DS, can have batch size more than 1

"""

import os
import copy
import numpy as np
import scipy.io as sio
import h5py
from torch.utils.data import Dataset
import torch
from utils.fastMRI.TorchDSTransforms import to_tensor

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class MRIMATTorchDS(Dataset):
    """Read MR Images from MAT files as Torch (custom) Dataset"""

    def __init__(self, root_path, transform=None):
        """
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to the OASIS1 standard
        Args:
            root_path (string): Directory with all the MAT files
        """        
        self.dataset = h5py.File(root_path, 'r') 
        self.transform = transform

    def __len__(self):
        """For returning the length of the file list"""
        return len(self.dataset)

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def __getitem__(self, idx):
        h5group =  self.dataset[str(idx)]
        sample = {}
        for item in h5group:
            sample[item] = h5group[item].value

        if self.transform is not None:
            sample = self.transform(sample)

        if type(sample['fully']) is not torch.Tensor:
            sample['fully'] = to_tensor(sample['fully'])
        if type(sample['under']) is not torch.Tensor:
            sample['under'] = to_tensor(sample['under'])
        sample['fileName'] = sample['fileName'].astype('U')[0]
        sample['subjectName'] = sample['subjectName'].astype('U')[0]

        return sample