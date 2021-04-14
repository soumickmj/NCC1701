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

import os
import shutil
import numpy as np
#import numpy.ma as ma
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.parallel
from skimage.measure import compare_ssim
import utils.MRITorchDSTransforms as transformsMRI
from Math.FrequencyTransforms import ifft2c, irfft2c, ifht2c
from Math.TorchFrequencyTransforms import ifft2cT

np.seterr(all='raise')

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
            data_real = np.take(data,[0],-1)
            data_imag = np.take(data,[1],-1)
            data = data_real + 1j * data_imag
            data = ifft2c(data) 
        elif(self.domain == 'real_fourier'):
            data_real = np.take(data,[0],-1)
            data_imag = np.take(data,[1],-1)
            data = data_real + 1j * data_imag
            data = irfft2c(data)
        elif(self.domain == 'hartley'):
            data = ifht2c(data)
        elif(self.domain == 'image'):
            return data
        else:
            print('Invalid Domain Specified or Domain specified cant be used to reconstruct image')
            sys.exit(0)
        if(np.iscomplex(data).any()):
            data = abs(data).astype('float32')
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

    def load_checkpoint(self, checkpoint_file=None, model=None, IsCuda=True):
        """Load a already saved checkpoint. Checkpoints are created after every epoch of training"""

        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            if IsCuda:
                checkpoint = torch.load(checkpoint_file)
         

            else:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
            last_epoch = checkpoint['epoch']
            model.net = checkpoint['model']
            model.net.load_state_dict(checkpoint['state_dict'])
            model.best_accuracy = checkpoint['best_accuracy']
            model.optimizer.load_state_dict(checkpoint['optimizer'])
            # if not IsCuda:
            #     if type(model.net) is nn.DataParallel:
            #         model.net = model.net.module
            if type(model.net) is nn.DataParallel:
                model.net = model.net.module
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
            return model, last_epoch
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))
            return None, None

    def load_checkpointGAN(self, checkpoint_file=None, model=None):
        """Load a already saved checkpoint, only for GAN. Checkpoints are created after every epoch of training"""

        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            last_epoch = checkpoint['epoch']
            model.netD = checkpoint['modelD']
            model.netG = checkpoint['modelG']
            model.netD.load_state_dict(checkpoint['state_dictD'])
            model.netG.load_state_dict(checkpoint['state_dictG'])
            model.best_accuracy = checkpoint['best_accuracy']
            model.optimizerD.load_state_dict(checkpoint['optimizerD'])
            model.optimizerG.load_state_dict(checkpoint['optimizerG'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
            return model, last_epoch
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))
            return None, None

    def load_checkpointCycleGAN(self, checkpoint_file=None, model=None):
        """Load a already saved checkpoint, only for GAN. Checkpoints are created after every epoch of training"""

        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            last_epoch = checkpoint['epoch']
            model.G_AB.load_state_dict(checkpoint['G_AB'])
            model.G_BA.load_state_dict(checkpoint['G_BA'])
            model.D_A.load_state_dict(checkpoint['D_A'])
            model.D_B.load_state_dict(checkpoint['D_B'])
            model.best_accuracy = checkpoint['best_accuracy']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
            return model, last_epoch
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))
            return None, None
    
    def save_checkpoint(self, state, is_best, path='', save_frequency=20):
        """Creates Checkpoints after every epoch of training. 
        Also, if the current epoch is the best so far, then it creates a copy of the current checkpoint as best"""

        if (not os.path.exists(path) and path is not ''):
            os.makedirs(path)
        if is_best:
            filename = os.path.join(path, 'checkpoint.pth.tar')
            torch.save(state, filename)
            filename_best = os.path.join(path, 'model_best.pth.tar')
            shutil.copyfile(filename, filename_best)
        else:            
            if(int(state['epoch'])%save_frequency == 0):
                filename = os.path.join(path, 'checkpoint.pth.tar')
                torch.save(state, filename)

    def adjust_learning_rate(self, optimizer, epoch, lrInitial, lrDecayNEpoch, lrDecayRate):
        """Sets the learning rate to the initial LR decayed by lrDecayRate every lrDecayNEpoch epochs"""

        lr = lrInitial * (lrDecayRate ** (epoch // lrDecayNEpoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def accuracy(self, output, target, tensorToNumpyTransformer):
        """Computes the accuracy (SSIM) of the generated (by generator) images
        tensorToNumpyTransformer = transformsMRI.FromTensorToNumpy3D() or transformsMRI.FromTensorToNumpy2D()"""

        with torch.no_grad():
            ssim = []
            for j in range(0, len(target)):
                #Out = abs(self.Domain2Image(transformTensorToNumpy3D(output.data[j]))).astype('float32')
                #Ref = abs(self.Domain2Image(transformTensorToNumpy3D(target[j]))).astype('float32')
                Out = np.squeeze(self.Domain2Image(tensorToNumpyTransformer(output.data[j])))
                Ref = np.squeeze(self.Domain2Image(tensorToNumpyTransformer(target[j])))
                if(np.iscomplex(Out).any()):
                    Out = abs(Out).astype('float32')
                    Ref = abs(Ref).astype('float32')
                try:
                    Out = (Out - Out.min())/(Out.max() - Out.min())
                    Ref = (Ref - Ref.min())/(Ref.max() - Ref.min())
                except:
                    try:
                        Out = Out / Out.max()
                        Ref = Ref / Ref.max()
                    except:
                        pass
                try:
                    ssim.append(compare_ssim(Ref, Out, data_range=Out.max() - Out.min(), multichannel=False))
                except:
                    ssim.append(-1)
            return ssim

    def torchFFT(self, tensor):
        tensor_k = torch.rfft(tensor[0,0,:,:], signal_ndim=2, normalized=True, onesided=False)
        tensor_k = tensor_k / abs(tensor_k).max()
        tensor_k.unsqueeze_(0).unsqueeze_(0)
        tensor_k = tensor_k.permute(0,1,4,2,3)
        tensor_k = tensor_k.view(tensor_k.size(0),tensor_k.size(1)*tensor_k.size(2),tensor_k.size(3),tensor_k.size(4))
        return tensor_k

    def torchIFFT(self, tensor_k):
        tensor_k = tensor_k.view(tensor_k.size(0),1,tensor_k.size(1),tensor_k.size(2),tensor_k.size(3))
        tensor_k = tensor_k.permute(0,1,3,4,2)
        tensor = torch.irfft(tensor[0,0,:,:,:], signal_ndim=2, normalized=True, onesided=False)
        tensor.unsqueeze_(0).unsqueeze_(0)
        return tensor

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


    #Function obtained online
    #def accuracy(self, output, target, topk=(1,)): This is a good accuracy function for non-regression problems
    #    """Computes the precision@k for the specified values of k"""
    #    with torch.no_grad():
    #        maxk = max(topk)
    #        batch_size = target.size(0)

    #        k, pred = output.topk(maxk, 1, True, True)
    #        #input (Tensor) – the input tensor <output>
    #        #k (int) – the k in “top-k” <maxk>
    #        #dim (int, optional) – the dimension to sort along <1>
    #        #largest (bool, optional) – controls whether to return largest or smallest elements <True> we set it to true as looking for largest, we should set it to False if we need smallest
    #        #sorted (bool, optional) – controls whether to return the elements in sorted order <True> 
    #        #out (tuple, optional) – the output tuple of (Tensor, LongTensor) that can be optionally given to be used as output buffers <_,pred> we can use either Tensor or LongTensor, we use the later
    #        pred = pred.t() #transposes dimensions 0 and 1
    #        tv=target.view(1, -1)
    #        tvx=tv.expand_as(pred)
    #        #correct = pred.eq(target.view(1, -1).expand_as(pred))

    #        res = []
    #        for k in topk:
    #            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #            res.append(correct_k.mul_(100.0 / batch_size))
    #        return res


class EvaluationParams(object):
    """Computes and stores the average and current value of various evaluation parameters such as accuracy, execution time etc."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the current paramter values"""

        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the current parameter value"""

        if(not isinstance(val,list)):
            val = [val]
        self.val += val
        self.sum = sum(self.val)
        self.count += n
        self.avg = self.sum / self.count