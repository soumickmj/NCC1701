#!/usr/bin/env python
#This is same as the orignal, but the interpolate + conv been replaced by convtrans plus the kernel size of convtrans is manupulative

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.TorchModule.padding import ReflectionPad3d

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ResidualBlock(nn.Module):
    def __init__(self, in_features, relu, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [  ReflectionPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        norm(in_features),
                        relu(),
                        nn.Dropout3d(p=0.2, inplace=True),
                        ReflectionPad3d(1),
                        nn.Conv3d(in_features, in_features, 3) ,
                        norm(in_features) 
                        ]


        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResNet(nn.Module):
    def __init__(self, n_channels=1, res_blocks=14, starting_nfeatures=32, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, forwardV=0, upinterp_algo='trilinear', input_shape=(150,256,256)): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(ResNet, self).__init__()

        self.forwardV = forwardV
        self.upinterp_algo = upinterp_algo
        if is_relu_leaky:
            relu = nn.PReLU
        else:
            relu = nn.ReLU
        if do_batchnorm:
            norm = nn.BatchNorm3d
        else:
            norm = nn.InstanceNorm3d

        in_channels = n_channels
        out_channels = n_channels
        # Initial convolution block
        intialConv = [   ReflectionPad3d(3),
                    nn.Conv3d(in_channels, starting_nfeatures, 7),
                    norm(starting_nfeatures),
                    relu() ]

        # Downsampling [need to save the shape for upsample]
        downsam = []
        output_sizes_down = [input_shape]
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            output_sizes_down.append([math.ceil(x/2) for x in output_sizes_down[-1]])
            downsam.append(nn.Sequential(  
                        nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                        norm(out_features),
                        relu()))
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            resblocks += [ResidualBlock(in_features, relu, norm)]

        # Upsampling
        upsam = []
        output_sizes_up = [output_sizes_down[-1]]
        out_features = in_features//2
        for _ in range(updown_blocks):
            kernel = ()
            output_sizes_up.append([math.ceil(x*2) for x in output_sizes_up[-1]])            
            for d in range(len(output_sizes_up[-1])):
                if output_sizes_up[-1][d] != output_sizes_down[-len(output_sizes_up)][d]:
                    kernel += (2,)
                else:
                    kernel += (3,)
            output_sizes_up[-1] = output_sizes_down[-len(output_sizes_up)]
            upsam.append(nn.Sequential(  
                        #ReflectionPad3d(1),
                        #nn.Conv3d(in_features, out_features, 3),
                        nn.ConvTranspose3d(in_features, out_features, kernel, stride=2, padding=1, output_padding=1),
                        norm(out_features),
                        relu()))
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [  ReflectionPad3d(3),
                        nn.Conv3d(starting_nfeatures, out_channels, 7),
                        nn.Sigmoid() ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.ModuleList(downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.ModuleList(upsam)
        self.finalconv = nn.Sequential(*finalconv)

        if forwardV == 6: #SuperResNet
            self.sliceup1 = nn.Sequential(  
                            ReflectionPad3d(1),
                            nn.Conv3d(starting_nfeatures, starting_nfeatures, 3),
                            norm(starting_nfeatures),
                            relu())

            self.sliceup2 = nn.Sequential(  
                            ReflectionPad3d(1),
                            nn.Conv3d(starting_nfeatures, starting_nfeatures, 3),
                            norm(starting_nfeatures),
                            relu())

    def forward(self, x, slthkns_src=None, slthkns_trgt=None):
        if self.forwardV == 0:
            #v0: Original Version
            x = self.intialConv(x)
            shapes = []
            for downblock in self.downsam:
                shapes.append(x.shape[2:])
                x = downblock(x)
            x = self.resblocks(x)
            for i, upblock in enumerate(self.upsam):
                #x = F.interpolate(x, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                x = upblock(x)
            return self.finalconv(x)

        elif self.forwardV == 1:
            #v1: input is added to the final output
            out = self.intialConv(x)
            shapes = []
            for downblock in self.downsam:
                shapes.append(out.shape[2:])
                out = downblock(out)
            out = self.resblocks(out)
            for i, upblock in enumerate(self.upsam):
                #out = F.interpolate(out, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                out = upblock(out)
            return x + self.finalconv(out)

        elif self.forwardV == 2:
            #v2: residual of v1 + input to the residual blocks added back with the output
            out = self.intialConv(x)
            shapes = []
            for downblock in self.downsam:
                shapes.append(out.shape[2:])
                out = downblock(out)
            out = out + self.resblocks(out)
            for i, upblock in enumerate(self.upsam):
                #out = F.interpolate(out, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                out = upblock(out)
            return x + self.finalconv(out)

        elif self.forwardV == 3:
            #v3: residual of v2 + input of the initial conv added back with the output
            out = x + self.intialConv(x)
            shapes = []
            for downblock in self.downsam:
                shapes.append(out.shape[2:])
                out = downblock(out)
            out = out + self.resblocks(out)
            for i, upblock in enumerate(self.upsam):
                #out = F.interpolate(out, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                out = upblock(out)
            return x + self.finalconv(out)

        elif self.forwardV == 4:
            #v4: residual of v3 + output of the initial conv added back with the input of final conv
            iniconv = x + self.intialConv(x)
            shapes = []
            if len(self.downsam) > 0:
                for i, downblock in enumerate(self.downsam):
                    if i == 0:
                        shapes.append(iniconv.shape[2:])
                        out = downblock(iniconv)
                    else:
                        shapes.append(out.shape[2:])
                        out = downblock(out)
            else:
                out = iniconv
            out = out + self.resblocks(out)
            for i, upblock in enumerate(self.upsam):
                #out = F.interpolate(out, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                out = upblock(out)
            out = iniconv + out
            return x + self.finalconv(out)

        elif self.forwardV == 5:
            #v5: residual of v4 + individual down blocks with individual up blocks
            outs = [x + self.intialConv(x)]
            shapes = []
            for i, downblock in enumerate(self.downsam):
                shapes.append(outs[-1].shape[2:])
                outs.append(downblock(outs[-1]))
            outs[-1] = outs[-1] + self.resblocks(outs[-1])
            for i, upblock in enumerate(self.upsam):
                outs[-1] = F.interpolate(outs[-1], shapes[-1-i], mode=self.upinterp_algo, align_corners=True) ##TODO: NEEEED TTOOOOO CHECCCKKK
                outs[-1] = outs[-2] + upblock(outs.pop())
            return x + self.finalconv(outs.pop())

        elif self.forwardV == 6:
            #v6: residual of v0 + SuperResNet interpolations

            nSlice_src = x.shape[2]
            nSlice_trgt = int((nSlice_src * slthkns_src) / slthkns_trgt)
            nSlice_trgt = 130 #TODO: Hotfix
            nSlice_missing = nSlice_trgt - nSlice_src #distribute the missing slices before down and after up (add a conv after the interpolations)

            x = self.intialConv(x)

            shape = (x.shape[2] + (nSlice_missing //2),) + x.shape[3:]
            x = F.interpolate(x, shape, mode=self.upinterp_algo, align_corners=True)
            x = self.sliceup1(x)

            shapes = []
            for downblock in self.downsam:
                shapes.append(x.shape[2:])
                x = downblock(x)
            x = self.resblocks(x)
            for i, upblock in enumerate(self.upsam):
                #x = F.interpolate(x, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                x = upblock(x)

            shape = x.shape[2:]
            shape = (x.shape[2] + (nSlice_missing - (nSlice_missing //2)),) + x.shape[3:]
            x = F.interpolate(x, shape, mode=self.upinterp_algo, align_corners=True)
            x = self.sliceup2(x)

            return self.finalconv(x)

        elif self.forwardV == 7:
            #v6: residual of v0 + SuperResNet interpolations

            nSlice_src = x.shape[2]
            nSlice_trgt = int((nSlice_src * slthkns_src) / slthkns_trgt)
            nSlice_trgt = 130 #TODO: Hotfix
            nSlice_missing = nSlice_trgt - nSlice_src #distribute the missing slices before down and after up (add a conv after the interpolations)

            x = self.intialConv(x)

            shape = (x.shape[2] + (nSlice_missing),) + x.shape[3:]
            x = F.interpolate(x, shape, mode=self.upinterp_algo, align_corners=True)
            x = self.sliceup1(x)

            shapes = []
            for downblock in self.downsam:
                shapes.append(x.shape[2:])
                x = downblock(x)
            x = self.resblocks(x)
            for i, upblock in enumerate(self.upsam):
                #x = F.interpolate(x, shapes[-1-i], mode=self.upinterp_algo, align_corners=True)
                x = upblock(x)

            shape = x.shape[2:]
            shape = (x.shape[2] + (nSlice_missing - (nSlice_missing //2)),) + x.shape[3:]
            x = F.interpolate(x, shape, mode=self.upinterp_algo, align_corners=True)
            x = self.sliceup2(x)

            return self.finalconv(x)