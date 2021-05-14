#!/usr/bin/env python

"""
This module is for creating MAT files from any given Custom Torch Dataset
The MAT files will be numbered with index numbers 

"""

import os
import shutil
import scipy.io as sio
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
        if sliceHandleType == '2DSelectMultiSlice' or sliceHandleType == '3DSelectSlice':
            self.startingSlNo = startingSelectSlice + 1
        else:
            self.startingSlNo = 0
    
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
                    newsample['subjectName'] = sample['subjectName']
                    for key in tensorKeys:
                        newsample[key] = sample[key][:,sl,...]
                    newsample['fileName'] = [sample['fileName'][0] + '_sl' + str(sl+self.startingSlNo).rjust(len(str(noOfSlice)), '0')]
                    #file_name = os.path.join(output_folder, str((i*noOfSlice)+sl+offset)+'.mat') #this way of calculating file name won't work if there are different slice thickness in different volumes
                    file_name = os.path.join(output_folder, str(combinedSliceCounter+sl+offset)+'.mat')
                    if len(os.listdir(output_folder)) != combinedSliceCounter+sl+offset:
                        print(file_name)
                    sio.savemat(file_name, newsample)
                combinedSliceCounter += noOfSlice
                pbar.update(1)