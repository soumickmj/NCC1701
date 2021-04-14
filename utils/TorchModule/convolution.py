import torch
import torch.nn as nn

class Conv2Plus1D(nn.Sequential):
    #Inspired by torchvision
    #kernel_size, stride, padding shoudl be supplied as int not tuple
    def __init__(self,
                 in_channels,
                 out_channels,
                 midplanes=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        if midplanes is None:
            midplanes = (in_channels * out_channels * kernel_size * kernel_size * kernel_size) // (in_channels * kernel_size * kernel_size + kernel_size * out_channels)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_channels, midplanes, kernel_size=(1, kernel_size, kernel_size),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=bias),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_channels, kernel_size=(kernel_size, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=bias))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)

class Conv2Plus0D(nn.Conv3d):
    #AKA Conv3DNoTemporal
    #Inspired by torchvision
    #kernel_size, stride, padding shoudl be supplied as int not tuple
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(Conv2Plus0D, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=bias)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)