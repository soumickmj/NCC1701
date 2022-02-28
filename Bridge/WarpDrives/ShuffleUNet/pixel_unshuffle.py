import torch.nn as nn

from . import icnr


class _double_conv_3d(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(_double_conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class _double_conv_2d(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(_double_conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def _pixel_unshuffle_3d(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:(C, rH, rW) to a
    tensor of shape :math:(*, r^2C, H, W).
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, depth, in_height, in_width = input.size()

    depth_final = depth // upscale_factor
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, depth_final, upscale_factor, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 3
    unshuffle_out = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return unshuffle_out.view(batch_size, channels, depth_final, out_height, out_width)

def _pixel_unshuffle_2d(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:(C, rH, rW) to a
    tensor of shape :math:(*, r^2C, H, W).
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnshuffle(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, bias=True, d=3):
        super(PixelUnshuffle, self).__init__()
        if d == 3:
            self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, bias=bias, padding=kernel//2)
            self.down_conv = _double_conv_3d(out_c*8, out_c, kernel, stride, bias)
            self.pu = _pixel_unshuffle_3d
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, bias=bias, padding=kernel//2)
            self.down_conv = _double_conv_2d(out_c*4, out_c, kernel, stride, bias)
            self.pu = _pixel_unshuffle_2d
        self.icnr_weights = icnr.ICNR(self.conv.weight, 2)
        self.conv.weight.data.copy_(self.icnr_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.down_conv(self.pu(x, 2))
        return x
