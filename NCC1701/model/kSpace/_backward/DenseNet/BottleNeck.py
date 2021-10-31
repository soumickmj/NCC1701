#!/usr/bin/env python

"""
This is a BottleNeckNet, to be used with fourier data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from utils.TorchAct.pelu import PELU3, PELU2, PELU1

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Not tested"

class BottleNeckNet(nn.Module):
    """description of class"""

    def __init__(self, input_size=256, bottle_neck_size=128, model_type='complexlinear', activation='prelu', normalization='instance'):
        #List of model types:
        #1. complexlinear: Complex input is flatnned and sent to the bottle-neck
        #2. simplelinear: Complex tensor sent as two seperate channels then flatenned and sent to the bootle-neck. Can also be used with hartley data
        #3. bilinear_complexlinear: Input complex tensor is sent as two seperate feature bilinear transofrm is applied, and output is sent to the bottle-neck. 
        #                           After bottle-neck data is sent to linear layer, from where data is returned similar to complexlinear
        #                           This is a theritical possibility. But couldn't even check the size with bottle_neck_size set to 2. So, practially not feasable.
        super().__init__()

        self.model_type = model_type

        if model_type=='complexlinear':
            self.fc1 = nn.Linear(input_size*input_size*2, bottle_neck_size*bottle_neck_size)
            self.fc2 = nn.Linear(bottle_neck_size*bottle_neck_size, input_size*input_size*2)
        elif model_type == 'simplelinear':
            self.fc1 = nn.Linear(input_size*input_size, bottle_neck_size*bottle_neck_size)
            self.fc2 = nn.Linear(bottle_neck_size*bottle_neck_size, input_size*input_size)
        elif model_type=='bilinear_complexlinear':
            self.fc1 = nn.Bilinear(input_size*input_size, input_size*input_size, bottle_neck_size*bottle_neck_size)
            self.fc2 = nn.Linear(bottle_neck_size*bottle_neck_size, input_size*input_size*2)

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'pelu3':
            self.activation = PELU3()
        elif activation == 'pelu2':
            self.activation = PELU2()
        elif activation == 'pelu1':
            self.activation = PELU1()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        #if normalization=='instance':
        #    self.norm = nn.InstanceNorm1d()
        #elif normalization=='batch':
        #    self.norm = nn.BatchNorm1d()

    def forward(self, x):
        original_shape=x.shape
        if self.model_type=='complexlinear':
            x = torch.flatten(x, 1,-1)
            out1 = self.fc1(x)
        elif self.model_type == 'simplelinear':
            x = torch.flatten(x, 2,-1)
            out1 = self.fc1(x)
        elif self.model_type=='bilinear_complexlinear':
            x = torch.flatten(x, 2,-1)
            out1 = self.fc1(x[:,0,...], x[:,1,...])
        #out1_act = self.activation(self.norm(out1))
        out1_act = self.activation(out1)
        out2 = self.fc2(out1_act)
        #out = self.activation(self.norm(out2))
        out = self.activation(out2)
        out = out.reshape(original_shape)
        return out 


