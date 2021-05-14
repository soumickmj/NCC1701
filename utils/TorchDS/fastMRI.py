#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of OASIS1

"""
import sys
import os
import sys
import random
from pathlib import Path
from glob import glob
import copy
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from utils.math.freq_trans import ifftNc_pyt, fftNc_pyt, fnorm_pyt
from utils.math.misc import minmax_complex_pyt, minmax
from utils.corruptors.fastMRIUnder import apply_mask as fastmriunder_apply_mask

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

#get file name from Column: file
acqType = ['AXFLAIR', 'AXT2', 'AXT1POST', 'AXT1PRE', 'AXT1'] #Column: acquisition
#coils 2 to 28 (only 7 with 2, 7 with 24, 2 with 28, 1 with 22, 26 with 18, mostly 16 and then 20 and then 16,14,12) (choose 16)
#slice 2 to 16 (only one with 2, only 5 with 10, 46 with 12, rest 14 and 16, mostly 16 (Choose 16)
fieldStrn = ['2,8936', '1,494'] #Column: zISMRM_systemFieldStrength_T
scanner = ['Aera', 'Avanto', 'Biograph_mMR', 'Prisma_fit', 'Skyra', 'TrioTim'] #Column: zISMRM_systemModel - all siemens
#1.5: Aera and Avanto

class MRITorchDS(Dataset):
    """Read MR Images (Under and Fully parallely) as Torch (custom) Dataset"""

    def __init__(self, **kwargs):
        """
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to the OASIS1 standard
        Args:
            root_fully (string): Directory with all the images - fully sampled.
            domain (string): In which domain network should work? Acceptable values: | fourier | hartley | image| compleximage  
            #TODO transform (callable, optional): Optional transform to be applied on each sample. It can be custom composed transform, or single pre-defined transform. TODO: Implement
            #TODO getROIRes (tuple, option): Either None if no ROI needs to be extracted, or else supply a tuple with size in terms of height and width
            filename_filter (fastMRI_ScannerModels) = ['Aera': 1.5T,
                            'Biograph_mMR': 3T,
                            'Prisma_fit': 3T,
                            'Skyra': 3T] 
                            Should be supplied as a list. This notifies which subsets to use 
            ds_split_xlsx (path to excel file): TODO: Explain
            corupt_type (Courruption Type): fastmriunder: if we use function of fastMRI, then the next parameter mask_func have to be supplied. mask_seed can be supplied if you want deterministic masks, or None if you want random for each
                                            maskunder: if we want to use our own mask mat file, then undersampling_mat param should supplied provide a mat file's content, a dictonary containing mask of different size
                                            motion: motion correction
            mask_func: explained in corupt_type (fastmriunder)
            mask_seed: explained in corupt_type (fastmriunder)
            #TODO undersampling_mask: explained in corupt_type (maskunder) 
            top_n_sampled: if set to None, then all volumes will be read. If set to any specific number, then only the top that many volumes will be read
        Both 'root_fully' and 'root_under' needs to follow above mentioned folder sturcture
        Files should be organized identically in both 'root_fully' and 'root_under'
        """     
        self.__dict__.update(kwargs) 
        self.file_list, _ = self.CreateImageBase(self.filename_filter, self.ds_split_xlsx, self.top_n_sampled) #Create a Image Base, which contains all the image paths in flie_listself.domain_transform 

        #TODO: make the params
        self.prenorm_ksp = True
        self.return_raw = True
        self.minmax_img = False
        
    def __len__(self):
        """For returning the length of the file list"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """For returning one sample from file_list
        at index idx
        after applying transform on them"""
        
        subject, fname, slice = self.file_list[idx]
        with h5py.File(fname, 'r') as data:
            fully = torch.from_numpy(data['kspace'][slice])
            slice_max_len = len(str(data['kspace'].shape[0]))
            target_fastmri = data['reconstruction_rss'][slice] 

        if self.prenorm_ksp:
            fully = fftNc_pyt(minmax_complex_pyt(ifftNc_pyt(fully, norm="ortho"), channel_wise=False), norm="ortho")     #TODO: param for channel_wise    
        
        if self. corupt_type =='fastmriunder':
            under, mask = fastmriunder_apply_mask(fully, mask_func=self.mask_func, seed=self.mask_seed)
        elif self. corupt_type =='maskunder':
            sys.exit("fastMRIDS: maskunder not implemented")
        elif self. corupt_type =='motion':
            sys.exit("fastMRIDS: motion not implemented")
        else:
            sys.exit("fastMRIDS: invalid corupt_type")

        if self.return_raw and self.domain != "fourier":
            sample = {
                'gt_ksp': fully,
                'inp_ksp': under,
            }
        else:
            sample = {}

        if self.domain == 'fourier':
            pass
        elif self.domain == 'hartley':
            sys.exit("fastMRIDS: hartley domain not implemented")
        elif self.domain == 'image':
            fully = torch.abs(ifftNc_pyt(fully,norm="ortho"))
            under = torch.abs(ifftNc_pyt(under,norm="ortho"))
            if self.minmax_img:
                fully = minmax(fully, channel_wise=False)
                under = minmax(under, channel_wise=False)
        elif self.domain == 'compleximage':
            fully = ifftNc_pyt(fully)
            under = ifftNc_pyt(under)
            if self.minmax_img:
                fully = minmax_complex_pyt(fully, channel_wise=False)
                under = minmax_complex_pyt(under, channel_wise=False)
        else:
            sys.exit("fastMRIDS: invalid domain")      
        
        ##TODO coil combine as a tranform
        sample = {
            **sample, 
            'gt': fully,
            'inp': under,
            'target_fastmri': target_fastmri,
            'mask': mask,
            'subjectName': subject,
            'fileName': fname.stem + '_' + str(slice).zfill(slice_max_len) 
            }        

        return sample

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def CreateImageBase(self, scannermodel_subset=None, ds_split_xlsx=None, top_n_sampled=None):
        """Creates a Image Base
        i.e. makes a file_list, which contains path of all the undersampled and fully sampled images, along with subject names and file names
        Returns completeFileList, a list of dictonaries, each corresponds to each pair of fully and undersampled images, each contains path2fully, path2under, fileName without extension, subjectName w.r.t. each image
        and also returns subjectNames (unique subject names)
        top_n_sampled: if None, then it reads all. If set to a value, then it returns only top that many volumes
        """
        if ds_split_xlsx is not None:
            df = pd.read_excel(ds_split_xlsx)

        completeFileList = []
        subjectNames = []
        files = list(Path(self.root_fully).iterdir())
        count = 0
        for fullpath_file_fully in sorted(files):
            filename = os.path.basename(fullpath_file_fully)
            subject = filename.split('.')[0]

            if ds_split_xlsx is not None:
                # search_tuple = df['file'].str.contains(filename) &  df['scannermodel'].str.contains('|'.join(scannermodel_subset))
                search_tuple = df['file'].str.contains(filename) &  df['zISMRM_systemModel'].str.contains('|'.join(scannermodel_subset)) #TODO: why and how zISMRM_systemModel
                if not search_tuple.any():
                    continue
                else:
                    # num_slices = df.loc[search_tuple, "nSlice"].values[0] #considering only one tuple should be returned
                    num_slices = df.loc[search_tuple, "nSlice"].values[0] #considering only one tuple should be returned
                
            #if ds_split_xlsx is not None:
            #    if not (df['file'].str.contains(filename) &  df['scannermodel'].str.contains('|'.join(ixi_subset))).any():
            #        continue
            #    else:
            #        num_slices = df.loc[df['file'].str.contains(filename) &  df['scannermodel'].str.contains('|'.join(ixi_subset)), "nslice"].values[0] #considering only one tuple should be returned
            
            #Create a dictonary for the current file, contains path2fully, path2under, fileName without extension, subjectName w.r.t. this image
            # completeFileList += [(subject, fullpath_file_fully, slice) for slice in range(10,num_slices-10)]
            completeFileList += [(subject, fullpath_file_fully, slice) for slice in range(num_slices)] #TODO:starting and ending slice
            if subject not in subjectNames: #create unique list of subject names
                subjectNames.append(subject)
            count += 1
            if top_n_sampled is not None and count == top_n_sampled:
                break
        # print(len(completeFileList))
        # print(subjectNames)
        return completeFileList, subjectNames