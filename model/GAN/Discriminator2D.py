#!/usr/bin/env python

"""
Discriminator of GAN (FirstGAN)
Uses 2D Images
Ends with a Fully connected convolution, followed by a fully connected linear layer
With static hyper parameters
"""

import torch.nn as nn

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing. Input channels in the last linear layer needs to be dynamic (currently static for input images 256x256x128). Hyperparameters needs to be dynamic"

class Discriminator(nn.Module):
    """Code for the Discriminator Network"""

    def __init__(self, n_channels = 1):
        """Constructor that will define the architecture of the discriminator"""

        super(Discriminator, self).__init__() # We inherit from the nn.Module tools.
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 4, 2, 1, bias = False), # We start with a convolution..
            #3 : in_channels - Input Size - becuase our image has 3 channels RGB
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
            nn.Dropout2d(p = 0.2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.
            #128: I/P Size, 128: out_channels, 4: Kernal Size, 2: Stride, 1: Padding
            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
            #We do BatchNorm becuase - ???
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Dropout2d(p = 0.2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(256), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Dropout2d(p = 0.2)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(512), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Dropout2d(p = 0.2)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution.
            #512: I/P Size, 1: out_channels (1-Real or 0-Fake), 4: Kernal Size, 1: Stride, 0: Padding
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
            )
        self.fc = nn.Sequential(
            nn.Linear(169, 1, bias = False), # Fully connected layer. Need to find a way to calculate this 845.
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
            )

    def forward(self, input): 
        """Function for Forward Propagation of the discriminator
        We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1
        Input will be either Fully sampled images (then output should ideally be 1) and output from generator (then output should ideally be 0)"""

        output = self.conv1(input) # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        #finalConvOPSize = output.size(1) * output.size(2) * output.size(3) * output.size(4)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output
