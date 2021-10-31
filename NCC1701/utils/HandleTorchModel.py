#!/usr/bin/env python

"""
This module helps to handle PyTorch Models
It can Save, Load Trained PyTorch Models
It can also calculates memory usage 
Note: It's advisable to use the model.Helper class for saving and loading trained models, because that class is more trailored as per the current application

"""

import os
import json
import torch
from Math.PyTorchModelSize import SizeEstimator

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def SaveModel(model, path):
    """
    Save a Trained PyTorch Custom Model (Including the network model, optimizer etc.)
    """

    if (not os.path.exists(path) and path is not ''):
        os.makedirs(path)
    state = {
            'model': model,
            'state_dict': model.net.state_dict(),
            'optimizer' : model.optimizer.state_dict(),
        }
    filename = os.path.join(path, 'model.pth.tar')
    torch.save(state, filename)
    
def LoadModel(path):
    """
    Load a Trained PyTorch Custom Model (Including the network model, optimizer etc.)
    """

    if os.path.isfile(path):
        print("=> loading saved model '{}'".format(path))
        checkpoint = torch.load(path)
        model = checkpoint['model']
        model.net.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded saved model")
        return model
    else:
        print("=> no checkpoint found at '{}'".format(path))
        return None
        

def GetModelMemoryUsage(input_shape, model):
    """
    Estimates the Memory usage of a PyTorch Model.
    Parameters:
    input_shape: complete input shape, including batch_size and number of channels
    model: model (netowrk model or even custom model with only one netowrk model) for which size to be estimated
    NOTE: This doesn't take input of custom model by default. But you need to supply a netowrk model.
    If you supply a custom model, then it will use the network model : model.net
    If your custom model has more than one netowork model (for example: netG and netD), this will not be able to automatically fetch both the netowrk models.
    """
    if(model.__class__.__bases__[0].__name__ == 'object'): #If custom model was supplied
        model = model.net #Consider the network model (model.net) from the supplied custom model

    se = SizeEstimator(model, input_size=input_shape)
    mb, bits = se.estimate_size()
    #print(se.param_bits) # bits taken up by parameters
    #print(se.forward_backward_bits) # bits stored for forward and backward
    #print(se.input_bits) # bits for input
    return mb, bits