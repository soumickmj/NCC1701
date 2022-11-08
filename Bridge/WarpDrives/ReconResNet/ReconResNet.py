#!/usr/bin/env python

import torch.nn as nn
from tricorder.torch.transforms import Interpolator

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class ResidualBlock(nn.Module):
    def __init__(self, in_features, drop_prob=0.2):
        super(ResidualBlock, self).__init__()

        conv_block = [layer_pad(1),
                      layer_conv(in_features, in_features, 3),
                      layer_norm(in_features),
                      act_relu(),
                      layer_drop(p=drop_prob, inplace=True),
                      layer_pad(1),
                      layer_conv(in_features, in_features, 3),
                      layer_norm(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownsamplingBlock, self).__init__()

        conv_block = [layer_conv(in_features, out_features, 3, stride=2, padding=1),
                      layer_norm(out_features),
                      act_relu()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, mode="convtrans", interpolator=None, post_interp_convtrans=False):
        super(UpsamplingBlock, self).__init__()

        self.interpolator = interpolator
        self.mode = mode
        self.post_interp_convtrans = post_interp_convtrans
        if self.post_interp_convtrans:
            self.post_conv = layer_conv(out_features, out_features, 1)

        if mode == "convtrans":
            conv_block = [layer_convtrans(
                in_features, out_features, 3, stride=2, padding=1, output_padding=1), ]
        else:
            conv_block = [layer_pad(1),
                          layer_conv(in_features, out_features, 3), ]
        conv_block += [layer_norm(out_features),
                       act_relu()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x, out_shape=None):
        if self.mode != "convtrans":
            return self.conv_block(self.interpolator(x, out_shape))
        if not self.post_interp_convtrans:
            return self.conv_block(x)
        x = self.conv_block(x)
        return (
            self.post_conv(self.interpolator(x, out_shape))
            if x.shape[2:] != out_shape
            else x
        )


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="sigmoid", forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False):  # should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256
        super(ResNet, self).__init__()

        layers = {}
        if is3D:
            layers["layer_conv"] = nn.Conv3d
            layers["layer_convtrans"] = nn.ConvTranspose3d
            layers["layer_norm"] = nn.BatchNorm3d if do_batchnorm else nn.InstanceNorm3d
            layers["layer_drop"] = nn.Dropout3d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad3d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad3d
            layers["interp_mode"] = 'trilinear'
        else:
            layers["layer_conv"] = nn.Conv2d
            layers["layer_convtrans"] = nn.ConvTranspose2d
            layers["layer_norm"] = nn.BatchNorm2d if do_batchnorm else nn.InstanceNorm2d
            layers["layer_drop"] = nn.Dropout2d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad2d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad2d
            layers["interp_mode"] = 'bilinear'
        layers["act_relu"] = nn.PReLU if is_relu_leaky else nn.ReLU
        globals().update(layers)

        self.forwardV = forwardV
        self.upinterp_algo = upinterp_algo

        interpolator = Interpolator(
            mode=layers["interp_mode"] if self.upinterp_algo == "convtrans" else self.upinterp_algo)

        # Initial convolution block
        intialConv = [layer_pad(3),
                      layer_conv(in_channels, starting_nfeatures, 7),
                      layer_norm(starting_nfeatures),
                      act_relu()]

        # Downsampling [need to save the shape for upsample]
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam.append(DownsamplingBlock(in_features, out_features))
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = [
            ResidualBlock(in_features, res_drop_prob) for _ in range(res_blocks)
        ]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam.append(UpsamplingBlock(in_features, out_features,
                         self.upinterp_algo, interpolator, post_interp_convtrans))
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [layer_pad(3),
                     layer_conv(starting_nfeatures, out_channels, 7), ]

        if out_act == "sigmoid":
            finalconv += [nn.Sigmoid(), ]
        elif out_act == "relu":
            finalconv += [act_relu(), ]
        elif out_act == "tanh":
            finalconv += [nn.Tanh(), ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.ModuleList(downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.ModuleList(upsam)
        self.finalconv = nn.Sequential(*finalconv)

        if self.forwardV == 0:
            self.forward = self.forwardV0
        elif self.forwardV == 1:
            self.forward = self.forwardV1
        elif self.forwardV == 2:
            self.forward = self.forwardV2
        elif self.forwardV == 3:
            self.forward = self.forwardV3
        elif self.forwardV == 4:
            self.forward = self.forwardV4
        elif self.forwardV == 5:
            self.forward = self.forwardV5

    def forwardV0(self, x):
        # v0: Original Version
        x = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(x.shape[2:])
            x = downblock(x)
        x = self.resblocks(x)
        for i, upblock in enumerate(self.upsam):
            x = upblock(x, shapes[-1-i])
        return self.finalconv(x)

    def forwardV1(self, x):
        # v1: input is added to the final output
        out = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV2(self, x):
        # v2: residual of v1 + input to the residual blocks added back with the output
        out = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV3(self, x):
        # v3: residual of v2 + input of the initial conv added back with the output
        out = x + self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV4(self, x):
        # v4: residual of v3 + output of the initial conv added back with the input of final conv
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
            out = upblock(out, shapes[-1-i])
        out = iniconv + out
        return x + self.finalconv(out)

    def forwardV5(self, x):
        # v5: residual of v4 + individual down blocks with individual up blocks
        outs = [x + self.intialConv(x)]
        shapes = []
        for downblock in self.downsam:
            shapes.append(outs[-1].shape[2:])
            outs.append(downblock(outs[-1]))
        outs[-1] = outs[-1] + self.resblocks(outs[-1])
        for i, upblock in enumerate(self.upsam):
            outs[-1] = upblock(outs[-1], shapes[-1-i])
            outs[-1] = outs[-2] + outs.pop()
        return x + self.finalconv(outs.pop())
