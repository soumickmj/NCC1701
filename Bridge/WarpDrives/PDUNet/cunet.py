# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
from torchcomplex import nn as cnn
import torch.nn.functional as F
import torchcomplex.nn.functional as cF


class CUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, depth=4, wf=6, padding=True,
                 batch_norm=True, up_mode='upsample', complex_weights=True, kernel_size=3):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(CUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm, complex_weights=complex_weights, kernel_size=kernel_size))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, complex_weights=complex_weights, kernel_size=kernel_size))
            prev_channels = 2**(wf+i)

        self.avg_pool2d = cnn.AvgPool2d(2)
        self.last = cnn.Conv2d(prev_channels, n_classes, kernel_size=1, complex_weights=complex_weights)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = self.avg_pool2d(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, complex_weights, kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(cnn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                               padding=int(padding), complex_weights=complex_weights))
        block.append(cnn.CReLU())
        if batch_norm:
            block.append(cnn.BatchNorm2d(out_size, complex_weights=complex_weights))

        block.append(cnn.Conv2d(out_size, out_size, kernel_size=kernel_size,
                               padding=int(padding), complex_weights=complex_weights))
        block.append(cnn.CReLU())
        if batch_norm:
            block.append(cnn.BatchNorm2d(out_size, complex_weights=complex_weights))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, complex_weights, kernel_size=3):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = cnn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2, complex_weights=complex_weights)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(cnn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                                    cnn.Conv2d(in_size, out_size, kernel_size=1, complex_weights=complex_weights))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, complex_weights=complex_weights, kernel_size=kernel_size)

    @staticmethod
    def embed_layer(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_x_0 = (target_size[1] - layer_width) // 2
        diff_x_1 = target_size[1] - layer_width - diff_x_0
        layer = F.pad(layer, (diff_x_0, diff_x_1, 0, 0), mode='circular')
        diff_y_0 = (target_size[0] - layer_height) // 2
        diff_y_1 = target_size[0] - layer_height - diff_y_0
        layer = F.pad(layer, (0, 0, diff_y_0, diff_y_1))
        return layer

    def forward(self, x, bridge):
        up = self.up(x)
        up = UNetUpBlock.embed_layer(up, bridge.shape[2:])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
