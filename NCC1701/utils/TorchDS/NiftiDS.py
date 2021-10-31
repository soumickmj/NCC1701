#!/usr/bin/env python

"""
This module is for creating a Custom Torch Dataset
This perticular module is created according to the folder sturcture of OASIS1

"""

import os
import sys
import random
from pathlib import Path
from glob import glob
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.HandleNifti import FileRead3D, HeaderRead
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

    def __init__(self, root_fully, root_under, fileExtension_under, domain='abs_fourier', transform=None, getROIMode=None, undersampling_mask=None, filename_filter = ['Guys','HH','IOP'], ds_split_xlsx = None, ds_split=None):
        """
        Initialize the MRI Torch Dataset Images has to be in Nifit format, folder structure is according to the OASIS1 standard
        Args:
            root_fully (string): Directory with all the images - fully sampled.
            root_under (string): Directory with all the images - under sampled.
            fileExtension_under (string): 
            domain (string): In which domain network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image | 
            transform (callable, optional): Optional transform to be applied on each sample. It can be custom composed transform, or single pre-defined transform
            filename_filter (ixi_subsets) = ['Guys': Guy’s Hospital using a Philips 1.5T (http://brain-development.org/scanner-philips-medical-systems-gyroscan-intera-1-5t/),
                            'HH': Hammersmith Hospital using a Philips 3T (http://brain-development.org/scanner-philips-medical-systems-intera-3t/),
                            'IOP': Institute of Psychiatry using a GE 1.5T (Scan params not available)] 
                            Should be supplied as a list. This notifies which subsets to use 
        Both 'root_fully' and 'root_under' needs to follow above mentioned folder sturcture
        Files should be organized identically in both 'root_fully' and 'root_under'
        """        
        self.root_fully = root_fully
        self.root_under = root_under
        self.fileExtension_under = fileExtension_under
        self.transform = transform
        self.file_list, _ = self.CreateImageBase(filename_filter, ds_split_xlsx, ds_split) #Create a Image Base, which contains all the image paths in flie_listself.domain_transform 
        self.domain = domain
        self.undersampling_mask = undersampling_mask #TODO: Can't work with different masks for each currently
        if(domain == 'fourier' or domain == 'fourier_magphase'): 
            self.domain_transform = lambda x: fnorm(fft2c(x)) #Matlab iquivalent: domain_transform = @fft2c
        elif(domain == 'real_fourier' or domain == 'real_fourier_magphase'): #Decreases the input size. Only preserves one half of the fourier as the other part is redundant
            self.domain_transform = lambda x: fnorm(rfft2c(x))
        elif(domain == 'abs_of_fourier'): #not acceptable
            self.domain_transform = lambda x: abs(fft2c(x))
        elif(domain == 'real_of_fourier'): #not acceptable
            self.domain_transform = lambda x: fft2c(x).real
        elif(domain == 'abs_of_real_fourier'): #not acceptable
            self.domain_transform = lambda x: abs(rfft2c(x))
        elif(domain == 'real_of_real_fourier'): #not acceptable
            self.domain_transform = lambda x: rfft2c(x).real
        elif(domain == 'hartley'):
            self.domain_transform = lambda x: hnorm(fht2c(x))
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
        if(self.domain == 'image'): #TODO: if other domains also uses undersampled images directly, then use the commented form
            #vol_under = self.domain_transform(FileRead3D(self.file_list[idx]['path2under']))
            vol_under = FileRead3D(self.file_list[idx]['path2under'])
        else:
            vol_under = ( vol_fully.transpose() * self.undersampling_mask.transpose() ).transpose()
            if(np.iscomplex(vol_fully).any()):
                if(self.domain == 'fourier_magphase' or self.domain == 'real_fourier_magphase'):
                    vol_fully=np.concatenate(f2mp(vol_fully), axis=-1)
                    vol_under=np.concatenate(f2mp(vol_under), axis=-1)
                else:
                    vol_fully=np.concatenate((np.real(vol_fully),np.imag(vol_fully)), axis=-1)
                    vol_under=np.concatenate((np.real(vol_under),np.imag(vol_under)), axis=-1)

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

    def GenerateTrainTestSet(self, xlsx_path, nTrain=None, nTest=None, nVal=None, percentTrain=0.5, percentTest = 0.5):
        #Only to be called from outside, to generate CSV with train and test test file names
        secure_random = random.SystemRandom()
        files = [x['subjectName']+'_'+x['fileName'] for x in self.file_list]
        if nTrain is None:
            nTrain = int(len(files) * percentTrain)
        if nTest is None:
            nTest = int(len(files) * percentTest)
        if nVal is None:
            nVal = len(files)-nTrain-nTest 

        train_set = secure_random.sample(files,k=nTrain)
        remaining = [x for x in files if x not in train_set]
        test_set = secure_random.sample(remaining,k=nTest)
        remaining = [x for x in remaining if x not in test_set]
        val_set = secure_random.sample(remaining,k=nVal)

        if len(val_set) == 0:
            dataset = {'train': train_set, 'test': test_set}
        else:
            dataset = {'train': train_set, 'test': test_set, 'val': val_set}

        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dataset.items() ]))
        df.to_excel(xlsx_path)

    def CreateImageBase(self, ixi_subset=None, ds_split_xlsx=None, ds_split=None):
        """Creates a Image Base
        i.e. makes a file_list, which contains path of all the undersampled and fully sampled images, along with subject names and file names
        Returns completeFileList, a list of dictonaries, each corresponds to each pair of fully and undersampled images, each contains path2fully, path2under, fileName without extension, subjectName w.r.t. each image
        and also returns subjectNames (unique subject names)
        """
        types = ('.img', '.nii', '.nii.gz') # the tuple of file types
        files = []
        for type in types:
            files.extend(glob(self.root_fully+'/**/*'+type, recursive=True))

        files.sort()
        completeFileList = [] #For storing a list of dictonaries, each corresponds to each pair of fully and undersampled images
        subjectNames = [] #For storing unique subject names
        
        if ds_split_xlsx is not None:
            df = pd.read_excel(ds_split_xlsx)

        for fullpath_file_fully in files:
            if(ixi_subset is not None and not any(subset in fullpath_file_fully for subset in ixi_subset)): #To ignore redundant files in OASIS DS
                continue
            fullpath_file_under = fullpath_file_fully.replace(self.root_fully, self.root_under)
            subject = os.path.basename(fullpath_file_fully).split('.')[0]

            file_name = os.path.basename(fullpath_file_under)
            index_of_dot = file_name.index('.')
            imagenameNoExt = file_name[:index_of_dot]
            fullpath_file_under = os.path.join(os.path.dirname(fullpath_file_under), imagenameNoExt)
            fullpath_file_under = glob(fullpath_file_under+'*')
            if len(fullpath_file_under)> 0:
                fullpath_file_under = fullpath_file_under[0]
            else:
                continue
            if(not(Path(fullpath_file_fully).is_file()) or not(Path(fullpath_file_under).is_file())): #check if both fully sampled and undersampled volmues are available, and only then read 
                continue

            if ds_split_xlsx is not None:
                search_string = subject+'_'+imagenameNoExt
                try:
                    if not df[ds_split].str.contains(search_string).any():
                        continue
                except KeyError:
                    print('mentioned ds_split is not present in the ds_split_xlsx')
                    break

            # header_fully = HeaderRead(fullpath_file_fully)

            # #Sanity check:
            # dim_fully = header_fully['dim']
            # if not (dim_fully[1:4] == [256,256,150]).all(): ##TODO HOTFIX. Send this values via param
            #     continue

            #Create a dictonary for the current file, contains path2fully, path2under, fileName without extension, subjectName w.r.t. this image
            currentFileDict = {
                'subjectName': subject,
                'fileName': imagenameNoExt,
                'path2fully': fullpath_file_fully,
                'path2under': fullpath_file_under
                }
            completeFileList.append(currentFileDict) #Add the dictonary corresponding to the file to the list of files
            if subject not in subjectNames: #create unique list of subject names
                subjectNames.append(subject)
        print(len(completeFileList))
        return completeFileList, subjectNames