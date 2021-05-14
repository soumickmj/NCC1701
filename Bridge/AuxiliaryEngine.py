#!/usr/bin/env python

"""
This is the Helper module for our custom model. 
Jobs:-
    Helper Class:-
    1. Spliting dataset into train and validate
    2. Save and load checkpoints (not dynamic for every model. Very much dependent for this ConvTrans)
    3. Adjust learning rate (never used yet)
    4. calculate accuracy (SSIM)(during validation)
    EvaluationParams Class:-
    1. For managing various evaluation parameters like accuracy, execution time etc. used in Model during training/validation

"""
import sys
import os
import shutil
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch
import torch.nn.parallel
from utils.math.freq_trans import ifftNc_np, ifftNc_pyt

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished for now. Other things can be added later"

class Helper(object):
    """This is a Helper class to our Model Class"""

    def __init__(self, domain):
        """Constructor of the Helper class. Initializes certain values"""              
        self.domain = domain

    def Domain2Image(self, data):
        if(self.domain == 'fourier'):
            data = ifftNc_np(data) 
        elif(self.domain == 'real_fourier'):
            data = np.abs(ifftNc_np(data))
        elif(self.domain == 'hartley'):
            sys.exit("Helper: hartley domain not ready")
            data = ifht2c(data)
        elif(self.domain == 'image'):
            return data
        else:
            print('Invalid Domain Specified or Domain specified cant be used to reconstruct image')
            sys.exit(0)
        if(np.iscomplex(data).any()):
            data = abs(data).astype('float32')
        return data

    def Domain2Image_pyt(self, data, return_abs=False):
        if self.domain == 'fourier' or self.domain == 'real_fourier':
            data = ifftNc_pyt(data,norm="ortho")
        elif(self.domain == 'hartley'):
            sys.exit("Helper: hartley domain not ready")
            data = ifht2c(data)
        elif self.domain == 'image' or self.domain == 'compleximage':
            pass
        else:
            print('Invalid Domain Specified or Domain specified cant be used to reconstruct image')
            sys.exit(0)
        if return_abs:
            return torch.abs(data)
        else:
            return data

    def GetTrainValidateLoader(self, dataset, batchSize, num_workers = 0, pin_memory=False, valid_percent = 0.25):
        """Divides given dataset to train and validation set, size of the validation set is defined using valid_percent parameter"""

        train_dataset = dataset.Copy()
        valid_dataset = dataset.Copy()

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_percent * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batchSize, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batchSize, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )

        return train_loader, valid_loader

    def load_checkpoint(self, checkpoint_file=None, model=None, device="cpu"):
        """Load a already saved checkpoint. Checkpoints are created after every epoch of training"""
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            try:
                checkpoint = torch.load(checkpoint_file, map_location=device)
                last_epoch = checkpoint['epoch']
                if 'model' in checkpoint and checkpoint['model'] is not None:
                    model.net = checkpoint['model']
                model.net.load_state_dict(checkpoint['state_dict'])
                model.best_accuracy = checkpoint['best_accuracy']
                model.optimizer.load_state_dict(checkpoint['optimizer'])
                model.scaler.load_state_dict(checkpoint['AMPScaler'])
                model.lrScheduler.load_state_dict(checkpoint['LRScheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file, checkpoint['epoch']))
                return model, last_epoch
            except:
                print("=> checkpoint found but corrupted at '{}'".format(checkpoint_file))
                return model, 0
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))
            return model, 0

    def save_checkpoint(self, state, is_best, path='', save_frequency=20, attempt=0):
        """Creates Checkpoints after every epoch of training. 
        Also, if the current epoch is the best so far, then it creates a copy of the current checkpoint as best"""

        if not os.path.exists(path) and path != '':
            os.makedirs(path)
        try:
            if is_best:
                filename = os.path.join(path, 'checkpoint.pth.tar')
                torch.save(state, filename)
                filename_best = os.path.join(path, 'model_best.pth.tar')
                shutil.copyfile(filename, filename_best)
            else:            
                if(int(state['epoch'])%save_frequency == 0):
                    filename = os.path.join(path, 'checkpoint.pth.tar')
                    torch.save(state, filename)
        except Exception as ex:
            if attempt == 0:
                del state['model']
                self.save_checkpoint(state, is_best, path, save_frequency, attempt=1)
                print('checkpoint saved after ignoring model')
            else:
                sys.exit(str(ex))

    def calculateMask(self, data):
        #TODO: think of 3D senarios
        #m = ma.masked_where(data != 0, data)
        #m = (m.mask.transpose() + m.mask.any(axis=1).transpose()).transpose()
        if(type(data) is torch.Tensor):
            data = data.numpy()
        m = ((np.ones(data.shape).transpose() * data.any(axis=1).transpose()).transpose()).astype(bool)
        undersamplingMask = m.astype(int)
        missingMask = (~m).astype(int)
        return undersamplingMask, missingMask

    def applyDataConsistancy(self, under, predicted, missingMask=None):
        #Apply data consistancy, undersampling mask calculated from "under" and then inverse of that mask applied on predicted
        if(missingMask is None):            
            corrected = torch.zeros(under.shape)
            for i in range(under.shape[0]):
                for j in range(under.shape[1]):
                    if(len(under.shape) == 5): #3D Images
                        for k in range(under.shape[2]):
                            slice_under = under[i,j,k,...]
                            slice_predicted = predicted[i,j,k,...]
                            undersamplingMask, missingMask = self.calculateMask(slice_under)
                            corrected[i,j,k,...] = slice_under + (slice_predicted * torch.from_numpy(missingMask).float())
                    else: #Assumed to have 4 dims (2D Images)
                        slice_under = under[i,j,...]
                        slice_predicted = predicted[i,j,...]
                        undersamplingMask, missingMask = self.calculateMask(slice_under)
                        corrected[i,j,...] = slice_under + (slice_predicted * torch.from_numpy(missingMask).float())
        else:
            corrected = under + (predicted.type(under.type()) * missingMask.type(under.type()))

        return corrected

class EvaluationParams(object):
    """Computes and stores the average and current value of various evaluation parameters such as accuracy, execution time etc."""

    def __init__(self, stat_mode='median'):
        self.reset()
        self.stat_mode = stat_mode

    def reset(self):
        """Resets the current paramter values"""

        self.val = []
        self.count = 0

    def update(self, val, n=None):
        """Update the current parameter value"""

        if(not isinstance(val,list)):
            val = [val]
        if n is None:
            n = len(val)
        self.val += val
        self.count += n

    def stat(self):
        if self.stat_mode == 'mean':
            return np.mean(self.val)
        elif self.stat_mode == 'median':
            return np.median(self.val)

    def sum(self): #For backward compatibility
        return sum(self.val)

    def avg(self): #For backward compatibility
        return self.sum / self.count