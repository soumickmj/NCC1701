import sys
import torch
import torch.nn as nn

from . import pixel_shuffle, pixel_unshuffle

# -------------------------------------------------------------------------------------------------------------------------------------------------##

class _double_conv(nn.Module):
    """
    Double Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True, conv_layer=nn.Conv3d):
        super(_double_conv, self).__init__()
        self.conv_1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv_2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu((x))
        x = self.conv_2(x)
        x = self.relu((x))

        return x


class _conv_decomp(nn.Module):
    """
    Convolutional Decomposition Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True, conv_layer=nn.Conv3d):
        super(_conv_decomp, self).__init__()
        self.conv1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv2 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv3 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv4 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu((x1))
        x2 = self.conv2(x)
        x2 = self.relu((x2))
        x3 = self.conv3(x)
        x3 = self.relu((x3))
        x4 = self.conv4(x)
        x4 = self.relu((x4))
        return x1, x2, x3, x4


class _concat(nn.Module):
    """
    Skip-Addition block
    """

    def __init__(self):
        super(_concat, self).__init__()

    def forward(self, e1, e2, e3, e4, d1, d2, d3, d4):
        self.X1 = e1 + d1
        self.X2 = e2 + d2
        self.X3 = e3 + d3
        self.X4 = e4 + d4
        x = torch.cat([self.X1, self.X2, self.X3, self.X4], dim=1)

        return x

# -------------------------------------------------------------------------------------------------------------------------------------------------##

class ShuffleUNet(nn.Module):

    def __init__(self, d=3, in_ch=1, num_features=64, n_levels=3, out_ch=1, kernel_size=3, stride=1):
        super(ShuffleUNet, self).__init__()

        self.n_levels = n_levels

        num_features = num_features
        filters = [num_features]
        for _ in range(n_levels):
            filters.append(filters[-1]*2)

        if d==3:
            conv_layer = nn.Conv3d
            ps_fact = (2 ** 2)
        elif d==2:
            conv_layer = nn.Conv2d
            ps_fact = 2
        else:
            sys.exit("Invalid d")

        # Input
        self.conv_inp = _double_conv(in_ch, filters[0], kernel_size, stride, conv_layer=conv_layer)     

        #Contraction path
        self.wave_down = nn.ModuleList()
        self.pix_unshuff = nn.ModuleList()
        self.conv_enc = nn.ModuleList()
        for i in range(0, n_levels):
            self.wave_down.append(_conv_decomp(filters[i], filters[i], kernel_size, stride, conv_layer=conv_layer))
            self.pix_unshuff.append(pixel_unshuffle.PixelUnshuffle(num_features * (2**i), num_features * (2**i), kernel_size, stride, d=d))       
            self.conv_enc.append(_double_conv(filters[i], filters[i+1], kernel_size, stride, conv_layer=conv_layer))     

        #Expansion path
        self.cat = _concat()
        self.pix_shuff = nn.ModuleList()
        self.wave_up = nn.ModuleList()
        self.convup = nn.ModuleList()
        for i in range(n_levels-1,-1,-1):
            self.pix_shuff.append(pixel_shuffle.PixelShuffle(num_features * (2**(i+1)), num_features * (2**(i+1)) * ps_fact, kernel_size, stride, d=d))
            self.wave_up.append(_conv_decomp(filters[i], filters[i], kernel_size, stride, conv_layer=conv_layer))
            self.convup.append(_double_conv(filters[i] * 5, filters[i], kernel_size, stride, conv_layer=conv_layer))

        #FC
        self.out = conv_layer(filters[0], out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        #Weight init
        for m in self.modules():
            if isinstance(m, conv_layer):
                weight = nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data.copy_(weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        encs = [self.conv_inp(x)]

        waves = []
        for i in range(self.n_levels):
            waves.append(self.wave_down[i](encs[-1]))
            _tmp = self.pix_unshuff[i](waves[-1][-1])
            encs.append(self.conv_enc[i](_tmp))
        
        dec = encs.pop()
        for i in range(self.n_levels):
            _tmp = self.pix_shuff[i](dec)
            _tmp_waves = self.wave_up[i](_tmp) + waves.pop()
            _tmp_cat = self.cat(*_tmp_waves)
            dec = self.convup[i](torch.cat([encs.pop(), _tmp_cat], dim=1))

        return self.out(dec)