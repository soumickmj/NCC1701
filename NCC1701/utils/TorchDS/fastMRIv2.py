import os
import numpy as np
import random
import math
from pathlib import Path
import pandas as pd
import h5py
from torch.utils.data import Dataset
from utils.MRITorchDSTransforms import CorruptByMotionNPMultiCoil, CorruptByMotionNPMultiCoilTorch 
import utils.fastMRI.TorchDSTransforms as transforms
from Math.FrequencyTransforms import ifft2c, fft2c

class MRITorchDS(Dataset):
    """Read MR Images from fastMRI Dataset as Torch (custom) Dataset"""

    def __init__(self, root_fully, domain='preimg_hartley', transform=None, getROIRes=None, filename_filter = None, ds_split_xlsx = None, corupt_type='fastmriunder', 
                 mask_func=None, mask_seed=None, undersampling_mat=None, top_n_sampled=None, n_center_slice=None, center_slice_percent=None, challenge=None, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.

            filename_filter: DSFiltersDict
        """
        if challenge not in ('singlecoil', 'multicoil', None):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.file_list = self.CreateImageBase(root_fully, ds_split_xlsx, filename_filter, n_center_slice, center_slice_percent, top_n_sampled, sample_rate)

        if domain.startswith('preimg_'):
            self.domain = domain.replace('preimg_','')
            self.preimg = True
        else:
            self.domain = domain
            self.preimg = False
        
        self.transform = transform #TODO
        self.getROIRes = getROIRes #TODO
        self.undersampling_mat = undersampling_mat 
        self. corupt_type = corupt_type
        self.mask_func = mask_func
        self.seed = mask_seed

        if corupt_type == 'motion':
            self.corrupter = CorruptByMotionNPMultiCoil(rotation_limit=5.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False) #TODO: Send these as param
        elif corupt_type == 'torchmotion':
            self.corrupter = CorruptByMotionNPMultiCoilTorch(rotation_limit=5.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normalize_rotation=False) #TODO: Send these as param

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        subject, fname, slice, nSlices = self.file_list[idx]
        with h5py.File(fname, 'r') as data:
            fully = data['kspace'][slice]
            #targetFastMRI = data[self.recons_key][slice] if self.recons_key in data else None #Unnecessary ATM
                        
            if self.preimg:
                #fully = np.transpose(fft2c(abs(ifft2c(np.transpose(fully, [1,2,0])))), [2,0,1])
                fully = abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fully))))
                if 'motion' not in self.corupt_type:
                    fully = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(fully)))

            #TODO Nicely
            fully = fully / abs(fully).max()
            if 'motion' not in self.corupt_type:
                fully = fully.astype(np.complex64)

            if self.corupt_type =='fastmriunder':
                fully = transforms.to_tensor(fully)
                under, mask = transforms.apply_mask(fully, mask_func=self.mask_func, seed=self.seed)
                corruptions = []
            elif self.corupt_type =='maskunder':
                mask = self.undersampling_mat['mask']
                under = fully * mask
                fully = transforms.to_tensor(fully)
                under = transforms.to_tensor(under)
                corruptions = []
            elif self.corupt_type =='motion':
                under, corruptions = self.corrupter(fully)
                fully = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(fully))).astype(np.complex64)
                under = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(under))).astype(np.complex64)
                fully = transforms.to_tensor(fully)
                under = transforms.to_tensor(under)
                mask = []
            elif self.corupt_type =='torchmotion':
                fully = transforms.to_tensor(fully)
                under, corruptions = self.corrupter(fully)
                fully = transforms.fft2(fully)
                under = transforms.fft2(under)
                mask = []
            else:
                import sys
                sys.exit()

            #If it is fourier_cmplxtnsr, then nothing has to be done
            if self.domain == 'fourier':
                fully = transforms.complex_to_channel(fully)
                under = transforms.complex_to_channel(under)
            elif self.domain == 'hartley':
                fully = transforms.fourier_to_hartley(fully)
                under = transforms.fourier_to_hartley(under)
            elif self.domain == 'image':
                fully = transforms.complex_abs(transforms.ifft2(fully))
                under = transforms.complex_abs(transforms.ifft2(under))
            elif self.domain == 'compleximage':
                fully = transforms.complex_to_channel(transforms.ifft2(fully))
                under = transforms.complex_to_channel(transforms.ifft2(under))
            elif self.domain == 'compleximage_cmplxtnsr':
                fully = transforms.ifft2(fully)
                under = transforms.ifft2(under)

            sample = {
                'fully': fully,
                'under': under,
                #'targetFastMRI': targetFastMRI, #Unnecessary ATM
                'mask': mask,
                'corruptions': corruptions,
                'subjectName': subject,
                'fileName': Path(fname).stem + '_' + str(slice).zfill(len(str(nSlices)))
                }

            return sample

    def Copy(self):
        """Creates a copy of the current object"""
        return copy.deepcopy(self)

    def CreateImageBase(self, root_fully, ds_split_xlsx, filters=None, n_center_slice=None, center_slice_percent=None, top_n_sampled=None, sample_rate=1):
        df = pd.read_excel(ds_split_xlsx)

        if filters is not None and filters != {}:
            dfFilter = None
            for k,v in filters.items():
                if dfFilter is None:
                    dfFilter = df[k].eq(v)
                else:
                    dfFilter &= df[k].eq(v)
            df = df[dfFilter]
                       
        if sample_rate < 1:
            df = df.sample(frac=sample_rate, replace=False, random_state=1701)
        
        filelist = []
        for index, row in df.iterrows():
            fname = os.path.join(root_fully, row['file'])
            if not os.path.isfile(fname):
                continue
            subject = row['file'].split('.')[0]
            nSlices = row['nSlice']
            slices = list(range(nSlices))

            if center_slice_percent is not None:
                n_center_slice = math.ceil(center_slice_percent * nSlices)
            if n_center_slice is not None:
                n_rejected = nSlices - n_center_slice
                slices = slices[(n_rejected//2):(n_rejected//2)+n_center_slice]
                
            filelist += [(subject, fname, slice, nSlices) for slice in slices]  
            
            if top_n_sampled is not None and index+1 == top_n_sampled:
                break

        return filelist