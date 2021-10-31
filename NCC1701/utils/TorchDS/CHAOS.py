#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of CHAOS

"""

import os
import sys
from pathlib import Path
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.HandleNifti import FileRead3D
from utils.GetROI import ClearNonROI
from Math.FrequencyTransforms import fft2c, rfft2c, fht2c, f2mp
from Math.Normalizations import fnorm, hnorm

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class MRITorchDS(Dataset):
    """Read MR Images (Under and Fully parallely) as Torch (custom) Dataset"""

    def __init__(self, root_fully, root_under=None, fileExtension_under=None, domain=None, transform=None, getROIMode=None, undersampling_mask=None):
        """
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to the OASIS1 standard
        Args:
            root_fully (string): Directory with all the images and masks.
            root_under (string): unused.
            fileExtension_under (string): unused
            domain (string): unused
            transform (callable, optional): Optional transform to be applied on each sample. It can be custom composed transform, or single pre-defined transform
        """        
        self.root_fully = root_fully
        self.transform = transform
        self.file_list, _ = self.CreateImageBase() #Create a Image Base, which contains all the image paths in flie_listself.domain_transform        
        
    def __len__(self):
        """For returning the length of the file list"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """For returning one sample from file_list
        at index idx
        after applying transform on them"""
        #(vol_fully, vol_under) = self.getROI(FileRead3D(self.file_list[idx]['path2fully']), FileRead3D(self.file_list[idx]['path2under']))

        vol_mask = FileRead3D(self.file_list[idx]['path2mask'])
        vol_image = FileRead3D(self.file_list[idx]['path2image'])
        vol_image = (vol_image - np.min(vol_image)) / (np.max(vol_image) - np.min(vol_image))

        sample = {
            'fully': vol_mask,
            'under': vol_image,
            'subjectName': self.file_list[idx]['subjectName'],
            'fileName': self.file_list[idx]['fileName']
            }

        if self.transform: #If transorm been supplied, apply it. It can be composed transform inlcuding custom transformations
            sample = self.transform(sample) 

        return sample

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def CreateImageBase(self):
        """Creates a Image Base
        i.e. makes a file_list, which contains path of all the undersampled and fully sampled images, along with subject names and file names
        Returns completeFileList, a list of dictonaries, each corresponds to each pair of fully and undersampled images, each contains path2fully, path2under, fileName without extension, subjectName w.r.t. each image
        and also returns subjectNames (unique subject names)
        """
        subfolders = [f for f in os.listdir(self.root_fully) if not os.path.isfile(os.path.join(self.root_fully, f))] #Get all subfolders (subject specific folders) from root
        subfolders.sort()
        completeFileList = [] #For storing a list of dictonaries, each corresponds to each pair of fully and undersampled images
        subjectNames = [] #For storing unique subject names
        for folder in subfolders:
            fullpath_subfolder = os.path.join(self.root_fully, folder) #creating 'fullpath_subfolder_fully' by concatenating 'root_fully' with subject specific folder
            subject = folder
            files = [f for f in os.listdir(fullpath_subfolder) if os.path.isfile(os.path.join(fullpath_subfolder, f))]
            files.sort()
            if (len(os.listdir(fullpath_subfolder)) > 0):
                for file in files:
                    if('mask' in file): #To ignore redundant files in OASIS DS
                        continue
                    fullpath_file_image = os.path.join(fullpath_subfolder, file)
                    imagenameNoExt = file.split('.')[0] #Extracting only the Image name, without extension
                    sequenceType = file.split('_')[0] #Extract sequence type, T1DUAL or T2SPIR
                    mask_filename = sequenceType + '_mask.nii'
                    fullpath_file_mask = os.path.join(fullpath_subfolder, mask_filename)
                    if(not(Path(fullpath_file_image).is_file()) or not(Path(fullpath_file_mask).is_file())): #check if both fully sampled and undersampled volmues are available, and only then read 
                        continue
                    #Create a dictonary for the current file, contains path2fully, path2under, fileName without extension, subjectName w.r.t. this image
                    currentFileDict = {
                        'subjectName': subject,
                        'fileName': imagenameNoExt,
                        'path2mask': fullpath_file_mask,
                        'path2image': fullpath_file_image
                        }
                    completeFileList.append(currentFileDict) #Add the dictonary corresponding to the file to the list of files
                    if subject not in subjectNames: #create unique list of subject names
                        subjectNames.append(subject)
        print(len(completeFileList))
        return completeFileList, subjectNames


