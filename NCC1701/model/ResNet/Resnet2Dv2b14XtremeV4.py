
#!/usr/bin/env python

"""


This version includes Dropout over Resnet2DSig.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
#from utils.TorchAct.pelu import PELU_oneparam as PELU

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        #PELU(),
                        nn.Dropout2d(p=0.2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]


        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        in_channels = n_channels
        out_channels = n_channels
        # Initial convolution block
        intialConv = [   nn.ReflectionPad2d(3),
                        nn.Conv2d(in_channels, starting_nfeatures, 7),
                        nn.InstanceNorm2d(starting_nfeatures),
                        nn.ReLU(inplace=True) ]

        # Downsampling
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            resblocks += [ResidualBlock(in_features)]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(starting_nfeatures, out_channels, 7),
                    nn.Sigmoid() ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.Sequential(*downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.Sequential(*upsam)
        self.finalconv = nn.Sequential(*finalconv)

    #def forward(self, x):
    #    #v1: input is added to the final output
    #    out = self.intialConv(x) #256
    #    out = self.downsam(out) #64
    #    out = self.resblocks(out) #64
    #    out = self.upsam(out) #256
    #    out = self.finalconv(out) #256
    #    return x + out #final residue, sending the initial input with the output

    #def forward(self, x):
    #    #v2: residual of v1 + input to the residual blocks added back with the output
    #    out = self.intialConv(x) #256
    #    out = self.downsam(out) #64
    #    out = out + self.resblocks(out) #64 - the input to the residual blocks is added back to the output
    #    out = self.upsam(out) #256
    #    out = self.finalconv(out) #256
    #    return x + out #final residue, sending the initial input with the output

    #def forward(self, x):
    #    #v3: residual of v2 + input of the final conv added back with the output = output becomes to have 64 feature maps
    #    #Not a possible option. As final conv's input is 64 feature maps, and output is 1, if we added them the final answer will have 64 feature maps which isn't correct
    #    out = self.intialConv(x) #256
    #    out = self.downsam(out) #64
    #    out = out + self.resblocks(out) #64 - the input to the residual blocks is added back to the output
    #    out = self.upsam(out) #256
    #    out = out + self.finalconv(out) #256 - the output of the upsampling block is added back to the output of the final convolution (I don't think it's needed)
    #    return x + out #final residue, sending the initial input with the output

    #def forward(self, x):
    #    #v3: residual of v2 + input of the initial conv added back with the output
    #    out = x + self.intialConv(x) #256 - 1 value of x added to 64 feature maps on initial conv
    #    out = self.downsam(out) #64
    #    out = out + self.resblocks(out) #64 - the input to the residual blocks is added back to the output
    #    out = self.upsam(out) #256
    #    out = self.finalconv(out) #256 - the output of the upsampling block is added back to the output of the final convolution (I don't think it's needed)
    #    return x + out #final residue, sending the initial input with the output

    def forward(self, x):
        #v4: residual of v3 + output of the initial conv added back with the input of final conv
        iniconv = x + self.intialConv(x) #256 - 1 value of x added to 64 feature maps on initial conv
        out = self.downsam(iniconv) #64
        out = out + self.resblocks(out) #64 - the input to the residual blocks is added back to the output
        out = iniconv + self.upsam(out) #256
        out = self.finalconv(out) #256 - the output of the upsampling block is added back to the output of the final convolution (I don't think it's needed)
        return out #final residue, sending the initial input with the output


