#!/usr/bin/env python

"""
This module is for creating MAT files from any given Custom Torch Dataset
The MAT files will be numbered with index numbers 

"""

import os
import scipy.io as sio
import shutil
from tqdm import tqdm
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

    def __init__(self, domain, IsRadial, sliceHandleType, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, getROIMode, undersampling_mask=None, filename_filter=None, splitXSLX=None, split2use=None):
        dsInitializer = TorchDSInitializer()
        self.dataloader, _ = dsInitializer.InitializeDS(domain, IsRadial, sliceHandleType, 1, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, 0, getROIMode, undersampling_mask, filename_filter, splitXSLX,split2use)
    
    def generateMATs(self, output_folder):
        if(not os.path.isdir(output_folder)):
            os.makedirs(output_folder)
            offset = 0
        else:
            print('The output folder specified already exists. Do you want to add the new MAT files to it? If no, then the existing files will be overwritten')
            answer = input("Enter Y/N: ")
            if answer == 'Y' or answer == 'y':
                offset = len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))]) #find out how many files already present in the folder and start naming the files from that
            else:
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)
                offset = 0
            
        with tqdm(total=len(self.dataloader.dataset)) as pbar:
            for i, sample in enumerate(self.dataloader, 0):
                for key, value in sample.items():
                    if type(value) is torch.Tensor:
                        sample[key] = (value[0,...]).numpy()
                file_name = os.path.join(output_folder, str(i+offset)+'.mat')
                sio.savemat(file_name, sample)
                pbar.update(1)