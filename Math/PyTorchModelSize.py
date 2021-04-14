#!/usr/bin/env python

"""
This module helps to estimate memory usage of any given PyTorch module.

This code is created with the help of pytorch_modelsize by jacobkimmel 
link: https://github.com/jacobkimmel/pytorch_modelsize

"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

class SizeEstimator(object):
    """Class to estimate size of a PyTorch model"""

    def __init__(self, model, input_size=(1,1,32,32), bits=32, flattened_layer = [-1]):
        """
        Estimates the size of PyTorch models in memory
        for a given input size
        """
        self.model = model
        self.input_size = input_size
        self.bits = bits
        self.flattened_layer = flattened_layer #-1 Means no flattened_layer. Or else, it is the layer number
        return

    def get_parameter_sizes(self):
        "Get sizes of all parameters in `model`"
        mods = list(self.model.children())
        #mods = list(self.model.modules()) #Got from original code
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        with torch.no_grad():
            input_ = Variable(torch.FloatTensor(*self.input_size))
            mods = list(self.model.children())
            #mods = list(self.model.modules()) #Got from original code
            out_sizes = []
            for i in range(0, len(mods)): 
            #for i in range(1, len(mods)): #Got from original code
                m = mods[i]
                if(i in self.flattened_layer):
                    input_ = input_.reshape(input_.size(0), -1)
                out = m(input_)
                out_sizes.append(np.array(out.size()))
                input_ = out

            self.out_sizes = out_sizes
            return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total