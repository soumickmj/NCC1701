import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.HandleNifti import FileRead3D

class NPDS(Dataset):

    def __init__(self, pathGT, pathIN, fileExtension_under=None, domain='hartley', transform=None, getROIMode=None, undersampling_mask=None, 
                 filename_filter = None, ds_split_xlsx = None, ds_split=None, contrasts=None, thickness=None):

        ##filename_filter: should be a tuple, where first element means the starting index and the second index should be the last element

        arrayIN = FileRead3D(pathIN).squeeze() #TODO: a way to send arrays directly
        arrayGT = FileRead3D(pathGT).squeeze()

        self.arrayIN = arrayIN
        self.arrayGT = arrayGT
        self.transform = transform

        self.CreateImageBase(filename_filter)
    
    def __len__(self): 
        return self.arrayIN.shape[-1]

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def __getitem__(self, i): 
        sample = {
            'fully': np.expand_dims(np.expand_dims(self.arrayGT[...,i],-1),-1),
            'under': np.expand_dims(np.expand_dims(self.arrayIN[...,i],-1),-1),
            'subjectName': str(i),
            'fileName': str(i)
            }

        if self.transform: #If transorm been supplied, apply it. It can be composed transform inlcuding custom transformations
            sample = self.transform(sample) 

        return sample

    def CreateImageBase(self, filename_filter=None, top_n_sampled=None):
        if filename_filter is not None:
            self.arrayIN = self.arrayIN[...,filename_filter[0]:filename_filter[1]]
            self.arrayGT = self.arrayGT[...,filename_filter[0]:filename_filter[1]]
        if top_n_sampled is not None:
            self.arrayIN = self.arrayIN[...,:top_n_sampled]
            self.arrayGT = self.arrayGT[...,:top_n_sampled]