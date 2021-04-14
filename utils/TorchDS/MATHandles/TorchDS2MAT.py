#!/usr/bin/env python

"""
This module is for creating MAT files from any given Custom Torch Dataset
The MAT files will be numbered with index numbers 

"""

import os
import scipy.io as sio
import torch
from utils.TorchDS.TorchDSInitializer import TorchDSInitializer

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

class TorchDS2MAT(object):
    """This class converts any Torch DS (especially Custom) to MAT files, saving them with their corresponding indices"""

    def __init__(self, domain, IsRadial, sliceHandleType, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, getROIMode, undersampling_mask=None):
        dsInitializer = TorchDSInitializer()
        self.dataloader, _ = dsInitializer.InitializeDS(domain, IsRadial, sliceHandleType, 1, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, 0, getROIMode, undersampling_mask)
    
    def generateMATs(self, output_folder):
        if(not os.path.isdir(output_folder)):
            os.makedirs(output_folder)
        for i, sample in enumerate(self.dataloader, 0):
            for key, value in sample.items():
                if type(value) is torch.Tensor:
                    sample[key] = (value[0,...]).numpy()
            file_name = os.path.join(output_folder, str(i)+'.mat')
            sio.savemat(file_name, sample)