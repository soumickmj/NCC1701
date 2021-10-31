import os
import sys
from pathlib import Path
import copy
import numpy as np
import xlrd 
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils.HandleDEN import ReadDENVol
from scipy.ndimage.measurements import label
from glob import glob

class AbdomenDS(Dataset):
    """description of class"""

    def __init__(self, datapath, mode):
        self.mode = mode
        self.listOfItems = glob(path+"/*")

    def __len__(self):
        """For returning the length of the dataset"""
        return len(self.listOfItems)

    def __getitem__(self, index):
        self.lstFilesDCM = glob(self.listOfItems[index]+"**/T1DUAL/DICOM_anon/InPhase/*")
        print(self.listOfItems[index])
        dicoms = []
        for ite in self.lstFilesDCM:
            RefDs = pydicom.read_file(ite)
            dicoms.append(RefDs.pixel_array)  
        dicoms = np.asarray(dicoms)
        dicoms = np.expand_dims(dicoms, 0)

        if(self.mode == "train"):                
            weight = self.cal_weight(dicoms)
            weight = np.asarray(weight)                
            ds = Data.TensorDataset(torch.from_numpy(dicoms/1.0).float(),torch.from_numpy(dicoms/1.0).float())
        else:
            ds = Data.TensorDataset(torch.from_numpy(dicoms/1.0).float())

        return ds


    def cal_weight(self, raw_data):
        shape = raw_data.shape
        #print("calculating weights.")
        dissim = self.numpack.zeros((shape[0],shape[1],shape[2],shape[3],shape[4],(config.radius-1)*2+1,(config.radius-1)*2+1))
        data = self.numpack.asarray(raw_data)
        padded_data = self.numpack.pad(data,((0,0),(0,0),(0,0),(config.radius-1,config.radius-1),(config.radius-1,config.radius-1)),'reflect')
        for m in range(2*(config.radius-1)+1):
            for n in range(2*(config.radius-1)+1):
                dissim[:,:,:,:,:,m,n] = data-padded_data[:,:,:,m:shape[3]+m,n:shape[4]+n]
        temp_dissim = self.numpack.exp(-self.numpack.power(dissim,2).sum(1,keepdims = True)/config.sigmaI**2)  
        dist = self.numpack.zeros((2*(config.radius-1)+1,2*(config.radius-1)+1))
        for m in range(1-config.radius,config.radius):
            for n in range(1-config.radius,config.radius):
                if m**2+n**2<config.radius**2:
                    dist[m+config.radius-1,n+config.radius-1] = self.numpack.exp(-(m**2+n**2)/config.sigmaX**2)
        print("weight calculated.")
        res = self.numpack.multiply(temp_dissim,dist)
        return res

    

