#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of OASIS1

"""

import os
import sys
from pathlib import Path
import copy
import numpy as np
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
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to Alex's 6 Channel Data
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

        vols_fully = []
        vols_under = []

        for p in self.file_list[idx]['paths2fully']:
            vols_fully.append(FileRead3D(p))
        for p in self.file_list[idx]['paths2under']:
            vols_under.append(FileRead3D(p))
        vol_fully = self.domain_transform(np.concatenate(vols_fully, axis=3))
        vol_under = self.domain_transform(np.concatenate(vols_under, axis=3))

        #Combine Channels

        if (vol_fully.max() == 0 or vol_under.max() == 0):
            #with open(r"Z:\Output\AlexT1T2\Attempt4-ImageMaskSeen-MinMax-ZeroGra-V1\insane.txt", "a") as myfile:
            #    myfile.write(str(self.file_list[idx]['paths2fully'][0]))
            return []

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

    def CreateImageBase(self, underSuffix='T1'):
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

            fullpaths_file_fully = []
            fullpaths_file_under = []

            imagenameNoExtUnderCh1 = imagenameNoExtUnder + '_First_channel'
            imagenameNoExtUnderCh2 = imagenameNoExtUnder + '_Second_channel'
            imagenameNoExtUnderCh3 = imagenameNoExtUnder + '_Third_channel'
            imagenameNoExtUnderCh4 = imagenameNoExtUnder + '_Fourth_channel'
            imagenameNoExtUnderCh5 = imagenameNoExtUnder + '_Fifth_channel'
            imagenameNoExtUnderCh6 = imagenameNoExtUnder + '_Sixth_channel'
            undersampledImageNameCh1 = imagenameNoExtUnderCh1+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh1))
            undersampledImageNameCh2 = imagenameNoExtUnderCh2+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh2))
            undersampledImageNameCh3 = imagenameNoExtUnderCh3+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh3))
            undersampledImageNameCh4 = imagenameNoExtUnderCh4+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh4))
            undersampledImageNameCh5 = imagenameNoExtUnderCh5+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh5))
            undersampledImageNameCh6 = imagenameNoExtUnderCh6+'.'+self.fileExtension_under 
            fullpaths_file_under.append(os.path.join(self.root_under, imagenameNoExtUnder, undersampledImageNameCh6))

            imagenameNoExtFullyCh1 = subject + '_First_channel'
            imagenameNoExtFullyCh2 = subject + '_Second_channel'
            imagenameNoExtFullyCh3 = subject + '_Third_channel'
            imagenameNoExtFullyCh4 = subject + '_Fourth_channel'
            imagenameNoExtFullyCh5 = subject + '_Fifth_channel'
            imagenameNoExtFullyCh6 = subject + '_Sixth_channel'
            fullyImageNameCh1 = imagenameNoExtFullyCh1+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh1))
            fullyImageNameCh2 = imagenameNoExtFullyCh2+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh2))
            fullyImageNameCh3 = imagenameNoExtFullyCh3+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh3))
            fullyImageNameCh4 = imagenameNoExtFullyCh4+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh4))
            fullyImageNameCh5 = imagenameNoExtFullyCh5+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh5))
            fullyImageNameCh6 = imagenameNoExtFullyCh6+'.'+self.fileExtension_under 
            fullpaths_file_fully.append(os.path.join(self.root_fully, imagenameNoExtFully, fullyImageNameCh6))

            #undersampledImageName = imagenameNoExtUnder+'.'+self.fileExtension_under 
            #fullpath_file_under = os.path.join(fullpath_subfolder_under, undersampledImageName)
            #Check if all paths are valid
            for p in fullpaths_file_fully:
                if not(Path(p).is_file()):
                    continue
            for p in fullpaths_file_under:
                if not(Path(p).is_file()):
                    continue
            #Create a dictonary for the current file, contains path2fully, path2under, fileName without extension, subjectName w.r.t. this image
            currentFileDict = {
                'subjectName': subject,
                'fileName': subject,
                'paths2fully': fullpaths_file_fully,
                'paths2under': fullpaths_file_under
                }
            completeFileList.append(currentFileDict) #Add the dictonary corresponding to the file to the list of files
            if subject not in subjectNames: #create unique list of subject names
                subjectNames.append(subject)
        print(len(completeFileList))
        return completeFileList, subjectNames


