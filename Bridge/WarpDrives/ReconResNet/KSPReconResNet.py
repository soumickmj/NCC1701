#!/usr/bin/env python

"""
KSpace version (Complex-valued convolution) of ReconResNet
"""
import torch
import torch.fft
import torch.nn as nn
import torchcomplex.nn as cnn
from tricorder.math.transforms.fourier import ifftNc_pyt
from tricorder.torch.transforms import Interpolator

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class ResidualBlock(nn.Module):
    def __init__(self, in_features, drop_prob=0.2, complex_weights=True):
        super(ResidualBlock, self).__init__()

        conv_block = [  # layer_pad(1),
                        layer_conv(in_features, in_features, 3,
                                   padding=1, complex_weights=complex_weights),
                        layer_norm(
                            in_features, complex_weights=complex_weights),
                        act_relu(),
                        layer_drop(p=drop_prob, inplace=True),
                        # layer_pad(1),
                        layer_conv(in_features, in_features, 3,
                                   padding=1, complex_weights=complex_weights),
                        layer_norm(in_features, complex_weights=complex_weights)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, complex_weights=True):
        super(DownsamplingBlock, self).__init__()

        conv_block = [layer_conv(in_features, out_features, 3, stride=2, padding=1, complex_weights=complex_weights),
                      layer_norm(
                          out_features, complex_weights=complex_weights),
                      act_relu()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, mode="convtrans", interpolator=None, post_interp_convtrans=False, complex_weights=True):
        super(UpsamplingBlock, self).__init__()

        self.interpolator = interpolator
        self.mode = mode
        self.post_interp_convtrans = post_interp_convtrans
        if self.post_interp_convtrans:
            self.post_conv = layer_conv(
                out_features, out_features, 1, complex_weights=complex_weights)

        if mode == "convtrans":
            conv_block = [layer_convtrans(
                in_features, out_features, 3, stride=2, padding=1, output_padding=1, complex_weights=complex_weights), ]
        else:
            conv_block = [  # layer_pad(1),
                            layer_conv(in_features, out_features, 3, padding=1, complex_weights=complex_weights), ]
        conv_block += [layer_norm(out_features, complex_weights=complex_weights),
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
    def __init__(self, n_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=0, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="relu", forwardV=0, upinterp_algo='sinc', post_interp_convtrans=False, is3D=False,
                 img_out_mode=0, fourier_norm_4imgout="ortho", under_replace=False, under_mask=None, inner_norm_ksp=True, complex_weights=True):
        # TODO: starting_nfeatures changed from 64 to 32, res_drop_prob from 0.2 to 0.0, is_relu_leaky from True to False, res_blocks from 14 to 7
        super(ResNet, self).__init__()

        self.img_out_mode = img_out_mode
        self.fourier_norm_4imgout = fourier_norm_4imgout
        self.under_replace = under_replace
        self.under_mask = under_mask
        self.missing_mask = None if under_mask is None else 1 - under_mask
        self.inner_norm_ksp = inner_norm_ksp

        layers = {}
        if is3D:
            layers["layer_conv"] = cnn.Conv3d
            layers["layer_convtrans"] = cnn.ConvTranspose3d
            layers["layer_norm"] = cnn.BatchNorm3d if do_batchnorm else nn.Identity
            layers["layer_drop"] = cnn.Dropout3d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad3d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad3d
        else:
            layers["layer_conv"] = cnn.Conv2d
            layers["layer_convtrans"] = cnn.ConvTranspose2d
            layers["layer_norm"] = cnn.BatchNorm2d if do_batchnorm else nn.Identity
            layers["layer_drop"] = cnn.Dropout2d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad2d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad2d
        layers["interp_mode"] = 'sinc'
        layers["act_relu"] = cnn.AdaptiveCmodReLU if is_relu_leaky else cnn.zReLU
        globals().update(layers)

        self.forwardV = forwardV
        self.upinterp_algo = upinterp_algo

        interpolator = Interpolator(
            mode=layers["interp_mode"] if self.upinterp_algo == "convtrans" else self.upinterp_algo)

        in_channels = n_channels
        out_channels = n_channels
        # Initial convolution block
        intialConv = [  # layer_pad(3),
                        layer_conv(in_channels, starting_nfeatures,
                                   7, padding=3, complex_weights=complex_weights),
                        layer_norm(starting_nfeatures,
                                   complex_weights=complex_weights),
                        act_relu()]

        # Downsampling [need to save the shape for upsample]
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam.append(DownsamplingBlock(
                in_features, out_features, complex_weights=complex_weights))
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = [
            ResidualBlock(
                in_features, res_drop_prob, complex_weights=complex_weights
            )
            for _ in range(res_blocks)
        ]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam.append(UpsamplingBlock(in_features, out_features,
                         self.upinterp_algo, interpolator, post_interp_convtrans, complex_weights=complex_weights))
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [  # layer_pad(3),
            layer_conv(starting_nfeatures, out_channels, 7, padding=3, complex_weights=complex_weights), ]

        if out_act == "sigmoid":
            finalconv += [cnn.Sigmoid(), ]
        elif out_act == "relu":
            finalconv += [act_relu(), ]
        elif out_act == "tanh":
            finalconv += [cnn.Tanh(), ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.ModuleList(downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.ModuleList(upsam)
        self.finalconv = nn.Sequential(*finalconv)

        if self.forwardV == 0:
            self.main_forward = self.forwardV0
        elif self.forwardV == 1:
            self.main_forward = self.forwardV1
        elif self.forwardV == 2:
            self.main_forward = self.forwardV2
        elif self.forwardV == 3:
            self.main_forward = self.forwardV3
        elif self.forwardV == 4:
            self.main_forward = self.forwardV4
        elif self.forwardV == 5:
            self.main_forward = self.forwardV5

    def forward(self, x):
        if self.inner_norm_ksp:
            factor = torch.abs(x).max()
            x = x / factor
        out = self.main_forward(x)

        if self.under_replace:
            out = self.missing_mask * out
            out = x + out

        if self.inner_norm_ksp:
            out *= factor

        if not self.img_out_mode:
            return out
        out = ifftNc_pyt(out, norm=self.fourier_norm_4imgout)
        if self.img_out_mode == 1:
            return out
        elif self.img_out_mode == 2:
            return torch.abs(out)

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
