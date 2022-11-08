import torch.nn as nn

from . import icnr


def _pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    See :class:`~torch.nn.PixelShuffle` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        # 1D example
        #>>> input = torch.Tensor(1, 4, 8)
        #>>> output = F.pixel_shuffle(input, 2)
        #>>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        #>>> input = torch.Tensor(1, 9, 8, 8)
        #>>> output = F.pixel_shuffle(input, 3)
        #>>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        #>>> input = torch.Tensor(1, 8, 16, 16, 16)
        #>>> output = F.pixel_shuffle(input, 2)
        #>>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)

class PixelShuffle(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, bias=True, d=3):
        super(PixelShuffle, self).__init__()
        if d==3:
            self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, bias=bias, padding=kernel//2)
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, bias=bias, padding=kernel//2)
        self.icnr_weights = icnr.ICNR(self.conv.weight, 2)
        self.conv.weight.data.copy_(self.icnr_weights)

    def forward(self, x):
        x = self.conv(x)
        x = _pixel_shuffle(x, 2)
        return x
