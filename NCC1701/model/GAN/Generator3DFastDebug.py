#!/usr/bin/env python

"""
Generator of GAN (FirstGAN)
This is a special FastDebug version with minimalistic architecture, just for testing/debuging other parts of code
Uses 3D Images
It's a Convolutional Translator. Without any size change from layer to layer.
Without Dropouts (being commented out)
With static hyper parameters
"""

import torch.nn as nn

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing. Hyperparameters needs to be dynamic"

class Generator(nn.Module):
    """Code for the Generator Network"""

    def __init__(self, n_channels = 1): 
        """Constructor that will define the architecture of the generator"""

        super(Generator, self).__init__() # We inherit from the nn.Module tools.
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_channels, 2, 3, 1, 1, bias = False), # We start with a convolution.
            #n_channels : in_channels - Input Size - becuase our image has 3 channels RGB
            #64: out_channels - No of Feature Maps in the Output
            #4: Kernal Size of 4x4
            #2: Stride
            #1: Padding
            #bias: setting it to false, by default true
            nn.LeakyReLU(0.2, inplace = True), # We apply a LeakyReLU.
            #ReLU(x)=max(0,x)
            #But, LeakyReLU(x)=max(0,x)+negative_slope*min(0,x)
            #0.2 : negative_slope
            #True: do the operation in-place. Be default false ?????
            #nn.Dropout3d(p = 0.2)
            )
        self.convFC = nn.Sequential(
            nn.Conv3d(2, n_channels, 3, 1, 1, bias = False), # We add another convolution.
            nn.Tanh(), # We apply another LeakyReLU.
            #nn.Dropout3d(p = 0.2)
            )

    def forward(self, input): 
        """Function for Forward Propagation of the generator
        We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output in terms of another image
        Input will be Undersampled Images and Output should be corrected (near fully Sampled) Images"""

        output = self.conv1(input)
        output = self.convFC(output)
        return output # We return the output containing the generated images.