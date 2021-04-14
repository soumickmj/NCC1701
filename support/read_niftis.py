#!/usr/bin/env python

"""
This module helps to read NIFTI files, following the OASIS1 Folder Stucture.
Folder structure should be : -

folder_path
    -fullpath_subfolder
        -files

'folder_path' is the root. under which contains individual folders for each subject ('fullpath_subfolder'). These subfolder names will be used as subject names.
Within these subfolders, should be MR volumes in Nifti format ('files') (.nii/.nii.gz/.img .hdr combo/.img)
For .img .hdr combo, '.hdr' files are ignored.

Contains two functions.
1. Reading Nifti Volumes from a given 'folder_path' following the above mentioned folder stucture
2. Reading Nifti Volumes from a given 'folder_path_fully' along with their corresponding undersampled volume from a given 'folder_path_under'. File extension should also be supplied for undersampled, as this can be different from fully sampled.


"""

import os
from pathlib import Path
import numpy as np
from utils.HandleNifti import FileRead

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


def ReadNIFTIVols(folder_path):
    """Function for readiong Reading Nifti Volumes from a given 'folder_path' following the above mentioned folder stucture
    Returns Volumes (as Nifiti 3D), subjectOfVol (subject ID (index in 'subjectNames') with respect to each Volume) , subjectNames (unique subject names), fileNames (fileNames with respect to each volume)
    """

    subfolders = [f for f in os.listdir(folder_path) if not os.path.isfile(os.path.join(folder_path, f))] #Get all subfolders (subject specific folders) from root
    subfolders.sort()
    Volumes = [] #For storing each volume after reading
    subjectNames = [] #For storing unique subject names
    subjectOfVol = [] #For storing subject ID with respect to each volume
    fileNames = [] #For storing file names with respect to each volume
    for folder in subfolders:
        fullpath_subfolder = os.path.join(folder_path, folder)
        subject = folder
        files = [f for f in os.listdir(fullpath_subfolder) if os.path.isfile(os.path.join(fullpath_subfolder, f))] #Get all volumes from a subject specific folder
        files.sort()
        if (len(os.listdir(folder_path)) > 0):
            for file in files:
                if (file.endswith('.hdr')): #.hdr files are ignored (as reading .img and .hdr for each volume is redundent)
                    continue
                fullpath_file = os.path.join(fullpath_subfolder, file)
                V = FileRead(fullpath_file) #Read nifit volume
                Volumes.append(V)
                if subject not in subjectNames: #create unique list of subject names
                    subjectNames.append(subject)
                subjectOfVol.append(subjectNames.index(subject)) #for subject ID with respect to each volume, use index in 'subjectNames' as ID
                fileNames.append(file.split('.')[0])
    Volumes = np.asarray(Volumes) #convert to NPArray from List
    return Volumes, subjectOfVol, subjectNames, fileNames

def ReadNIFTIVolsWithUnder(folder_path_fully, folder_path_under, extension_under):
    """Reading Nifti Volumes from a given 'folder_path_fully' along with their corresponding undersampled volume from a given 'folder_path_under'. File extension ('extension_under') should also be supplied for undersampled, as this can be different from fully sampled.
    Both 'folder_path_fully' and 'folder_path_under' needs to follow above mentioned folder sturcture
    Files should be organized identically in both 'folder_path_fully' and 'folder_path_under'
    Returns Volumes (as Nifiti 3D) (Fully sampled volume), VolumesUnder (as Nifiti 3D) (Undersampled volume with respect to each Fully Sampled Volume), subjectOfVol (subject ID (index in 'subjectNames') with respect to each Volume) , subjectNames (unique subject names), fileNames (fileNames with respect to each volume)
    """

    subfolders_fully = [f for f in os.listdir(folder_path_fully) if not os.path.isfile(os.path.join(folder_path_fully, f))] #Get all subfolders (subject specific folders) from root
    subfolders_fully.sort()
    Volumes = [] #For storing each fully sampled volume after reading
    VolumesUnder = [] #For storing each under sampled volume after reading with respect to each fully sampled volume
    subjectNames = [] #For storing unique subject names
    subjectOfVol = [] #For storing subject ID with respect to each volume
    fileNames = [] #For storing file names with respect to each volume
    for folder in subfolders_fully:
        fullpath_subfolder_fully = os.path.join(folder_path_fully, folder) #creating 'fullpath_subfolder_fully' by concatenating 'folder_path_fully' with subject specific folder
        fullpath_subfolder_under = os.path.join(folder_path_under, folder) #creating 'fullpath_subfolder_under' by concatenating 'folder_path_under' with subject specific folder
        subject = folder
        files = [f for f in os.listdir(fullpath_subfolder_fully) if os.path.isfile(os.path.join(fullpath_subfolder_fully, f))] #Get all volumes from a subject specific folder (considering only fully, file names should be same in under as well)
        files.sort()
        if (len(os.listdir(fullpath_subfolder_fully)) > 0):
            for file in files:
                if (file.endswith('.hdr')):  #.hdr files are ignored (as reading .img and .hdr for each volume is redundent)
                    continue
                fullpath_file_fully = os.path.join(fullpath_subfolder_fully, file)
                imagenameNoExt = file.split('.')[0] #Extracting only the Image name, without extension
                undersampledImageName = imagenameNoExt+'.'+extension_under #Creating file name for undersampled volume. By combining image name of fully without extension, dot, and extension of undersampled volumes
                fullpath_file_under = os.path.join(fullpath_subfolder_under, undersampledImageName)
                if(not(Path(fullpath_file_fully).is_file()) or not(Path(fullpath_file_under).is_file())): #check if both fully sampled and undersampled volmues are available, and only then read 
                    continue
                V = FileRead(fullpath_file_fully) #Read fully sampled nifit volume
                Vu = FileRead(fullpath_file_under) #Read undersampled nifit volume
                Volumes.append(V)
                VolumesUnder.append(Vu)
                if subject not in subjectNames: #create unique list of subject names
                    subjectNames.append(subject)
                subjectOfVol.append(subjectNames.index(subject)) #for subject ID with respect to each volume, use index in 'subjectNames' as ID
                fileNames.append(imagenameNoExt)
    Volumes = np.asarray(Volumes) #convert to NPArray from List
    VolumesUnder = np.asarray(VolumesUnder)
    return Volumes, VolumesUnder, subjectOfVol, subjectNames, fileNames                