#!/usr/bin/env python

"""
This module is for initializing Custom Torch Dataset
This will be called from Model.py and also from TorchDS2MAT.py, may be from other places in future
Except for MRIMATTorchDS, for that it is directly called from Model.py without initializer

"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils.MRITorchDSTransforms as transformsMRI

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

class TorchDSInitializer(object):
    """This class Initializes any given Custom Torch DS. This has been called from Model and also other palces (like, TorchDS2MAT) for initializing.
    The Initialize function of this class returns dataset and dataloader"""

    def InitializeDS(self, domain, sliceHandleType, batchSize, pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, 
                           folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode, undersampling_mask=None, 
                           filename_filter=None, splitXSLX=None, split2use=None, corupt_type='fastmriunder', mask_func=None, 
                           mask_seed=None, top_n_sampled=None, transform=None, isRadial=False):
        if transform:
            if type(transform) is list:
                listOfTransforms = transform 
            else:
                listOfTransforms = [transform]
        else:
            listOfTransforms = []
                
        if(domain == 'image'):
            listOfTransforms += [transformsMRI.MinMaxNormalization(), ]
        # else:
        #     listOfTransforms += [transformsMRI.ToTensor3D(),]

        # if(sliceHandleType == '2DSingleSlice'):
        #     listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=sliceHandleType,sliceno=sliceno)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        # elif(sliceHandleType == '2DSelectMultiSlice' or sliceHandleType == '3DSelectSlice'):
        #     listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=sliceHandleType, startingSelectSlice=startingSelectSlice, endingSelectSlice=endingSelectSlice)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        # elif(sliceHandleType == '2DMultiSlice'):
        #     listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=sliceHandleType)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        
        transform = transforms.Compose(listOfTransforms)
        
        # Loading the dataset
        dataset = dsClass(root_fully=folder_path_fully, root_under=folder_path_under, extension_under=extension_under, domain=domain,
                            transform=transform, getROIRes=getROIMode, undersampling_mask=undersampling_mask, filename_filter=filename_filter,
                            ds_split_xlsx=splitXSLX, ds_split=split2use, corupt_type=corupt_type, mask_func=mask_func, mask_seed=mask_seed,
                            top_n_sampled=top_n_sampled)
        dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = num_workers, pin_memory = pin_memory)

        return dataloader, dataset