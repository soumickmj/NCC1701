#!/usr/bin/env python

"""
This module is for creating MAT files from any given Custom Torch Dataset
The MAT files will be numbered with index numbers 

"""

import os
import numpy as np
import h5py
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

class TorchDS2H5(object):
    """This class converts any Torch DS (especially Custom) to MAT files, saving them with their corresponding indices"""

    def __init__(self, domain, IsRadial, sliceHandleType, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, getROIMode, undersampling_mask=None, filename_filter=None, splitXSLX=None, split2use=None):
        dsInitializer = TorchDSInitializer()
        self.dataloader, _ = dsInitializer.InitializeDS(domain, IsRadial, sliceHandleType, 1, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, 0, getROIMode, undersampling_mask, filename_filter, splitXSLX,split2use)
        if sliceHandleType == '2DSelectMultiSlice' or sliceHandleType == '3DSelectSlice':
            self.startingSlNo = startingSelectSlice + 1
        else:
            self.startingSlNo = 0
    
    def generateMATs(self, output_file):
        if(not os.path.isfile(output_file)):
            h5 = h5py.File(output_file, "w")
            offset = 0
        else:
            print('The HDF5 file specified already exists. Do you want to append to it? If no, then the existing file will be overwritten')
            answer = input("Enter Y/N: ")
            if answer == 'Y' or answer == 'y':
                h5 = h5py.File(output_file, "a")
                offset = len(h5)
            else:
                h5 = h5py.File(output_file, "w")
                offset = 0
        noOfSlice = 0
        combinedSliceCounter = 0 #counts already stored slice conters (of previous volumes)
        with tqdm(total=len(self.dataloader.dataset)) as pbar:
            for i, sample in enumerate(self.dataloader, 0):
                tensorKeys = []
                nonTensorKeys = []
                for key, value in sample.items():
                    if type(value) is torch.Tensor:
                        sample[key] = (value[0,...]).numpy()
                        noOfSlice = sample[key].shape[1]
                        tensorKeys.append(key)
                    else:
                        nonTensorKeys.append(key)
                for sl in range(noOfSlice): 
                    newsample = {}
                    newsample['subjectName'] = np.array(sample['subjectName'])
                    for key in tensorKeys:
                        newsample[key] = sample[key][:,sl,...]
                    newsample['fileName'] = np.array([sample['fileName'][0] + '_sl' + str(sl+self.startingSlNo).rjust(len(str(noOfSlice)), '0')])
                    if len(h5) != combinedSliceCounter+sl+offset:
                        print(file_name)

                    h5ds = h5.create_group(str(combinedSliceCounter+sl+offset))
                    for k, v in newsample.items():
                        if v.dtype.type is np.str_:
                            v = v.astype('S')
                        h5ds.create_dataset(k,data=v)

                combinedSliceCounter += noOfSlice
                pbar.update(1)