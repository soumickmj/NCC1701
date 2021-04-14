#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of OASIS1

"""

import os
import sys
from pathlib import Path
import copy
from torch.utils.data import Dataset, DataLoader
from utils.HandleNifti import FileRead3D
from utils.GetROI import ClearNonROI
from Math.FrequencyTransforms import fft2c, rfft2c, fht2c

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class MRITorchDS(Dataset):
    """Read MR Images (Under and Fully parallely) as Torch (custom) Dataset"""

    def __init__(self, root_fully, root_under, fileExtension_under, domain='abs_fourier', transform=None, getROIMode=None):
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
        self.root_fully = root_fully
        self.root_under = root_under
        self.fileExtension_under = fileExtension_under
        self.fileExtension_fully = fileExtension_under ##########TODO: seperate for fully
        self.transform = transform
        self.file_list, _ = self.CreateImageBase() #Create a Image Base, which contains all the image paths in flie_listself.domain_transform        
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
        return len(self.file_list)

    def __getitem__(self, idx):
        """For returning one sample from file_list
        at index idx
        after applying transform on them"""
        #(vol_fully, vol_under) = self.getROI(FileRead3D(self.file_list[idx]['path2fully']), FileRead3D(self.file_list[idx]['path2under']))

        vol_fully = self.domain_transform(FileRead3D(self.file_list[idx]['path2fully']))
        vol_under = self.domain_transform(FileRead3D(self.file_list[idx]['path2under']))

        sample = {
            'fully': vol_fully,
            'under': vol_under,
            'subjectName': self.file_list[idx]['subjectName'],
            'fileName': self.file_list[idx]['fileName']
            }

        if self.transform: #If transorm been supplied, apply it. It can be composed transform inlcuding custom transformations
            sample = self.transform(sample) 

        return sample

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def CreateImageBase(self, underSuffix='T1', onlyBrain=True):
        """Creates a Image Base
        i.e. makes a file_list, which contains path of all the undersampled and fully sampled images, along with subject names and file names
        Returns completeFileList, a list of dictonaries, each corresponds to each pair of fully and undersampled images, each contains path2fully, path2under, fileName without extension, subjectName w.r.t. each image
        and also returns subjectNames (unique subject names)
        """
        subfolders_fully = [f for f in os.listdir(self.root_fully) if not os.path.isfile(os.path.join(self.root_fully, f))] #Get all subfolders (subject specific folders) from root
        subfolders_fully.sort()
        completeFileList = [] #For storing a list of dictonaries, each corresponds to each pair of fully and undersampled images
        subjectNames = [] #For storing unique subject names
        for folder in subfolders_fully:
            subject = folder
            imagenameParts = subject.split('-')
            imagenameNoExtUnder = imagenameParts[0] + '-' + imagenameParts[1] + '-' + imagenameParts[2] + '-' + underSuffix
            
            if(onlyBrain):
                fullyImageName = subject+'_brain.'+self.fileExtension_fully #Creating file name for fully volume. By combining image name of fully without extension, dot, and extension of undersampled volumes
                fullpath_file_fully = os.path.join(self.root_fully, subject, fullyImageName)
                undersampledImageName = imagenameNoExtUnder+'_brain.'+self.fileExtension_under #Creating file name for undersampled volume. By combining image name of fully without extension, dot, and extension of undersampled volumes
                fullpath_file_under = os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageName)
            else:
                fullyImageName = subject+'.'+self.fileExtension_fully #Creating file name for fully volume. By combining image name of fully without extension, dot, and extension of undersampled volumes
                fullpath_file_fully = os.path.join(self.root_fully, subject, fullyImageName)
                undersampledImageName = imagenameNoExtUnder+'.'+self.fileExtension_under #Creating file name for undersampled volume. By combining image name of fully without extension, dot, and extension of undersampled volumes
                fullpath_file_under = os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageName)
                                    
            if(not(Path(fullpath_file_fully).is_file()) or not(Path(fullpath_file_under).is_file())): #check if both fully sampled and undersampled volmues are available, and only then read 
                continue

            #Create a dictonary for the current file, contains path2fully, path2under, fileName without extension, subjectName w.r.t. this image
            currentFileDict = {
                'subjectName': subject,
                'fileName': subject,
                'path2fully': fullpath_file_fully,
                'path2under': fullpath_file_under
                }
            completeFileList.append(currentFileDict) #Add the dictonary corresponding to the file to the list of files
            if subject not in subjectNames: #create unique list of subject names
                subjectNames.append(subject)
        print(len(completeFileList))
        return completeFileList, subjectNames


