import torch
import torch.nn.functional as F
from torch import nn


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(layer_conv(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(layer_batchnorm(out_size))

        block.append(layer_conv(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(layer_batchnorm(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = layer_convtrans(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode=interp_mode, scale_factor=2),
                                    layer_conv(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1]), diff_x:(diff_x + target_size[2])]
        #  _, _, layer_height, layer_width = layer.size() #for 2D data
        # diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2
        # return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        up = F.interpolate(up, size=bridge.shape[2:], mode=interp_mode)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

class UNet(nn.Module):
    """
    Implementation of
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    https://arxiv.org/abs/1505.04597

    Using the default arguments will yield the exact version used
    in the original paper

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
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
        droprate (float): Rate of dropout. If undesired, then 0.0
        is3D (bool): If a 3D or 2D version of U-net
        returnBlocks (bool) : If True, it will return the blocks created during downPath. If downPath is False, then it will be ignored
        downPath and upPath (bool): If only the downpath or uppath of the U-Net is needed, make the other one False

    Forward call:
        x (Tensor): Input Tensor
        blocks (list of Tensors): If only upPath is set to True, then this will be used during the forward of the uppath. If not desired, then supply blank list
    """
    def __init__(self, in_channels=1, out_channels=1, depth=3, wf=6, padding=True,
                 batch_norm=False, up_mode='upconv', droprate=0.0, is3D=False, 
                 returnBlocks=False, downPath=True, upPath=True):
        super(UNet, self).__init__()
        layers = {}
        if is3D:
            layers["layer_conv"] = nn.Conv3d
            layers["layer_convtrans"] = nn.ConvTranspose3d
            layers["layer_batchnorm"] = nn.BatchNorm3d
            layers["layer_drop"] = nn.Dropout3d
            layers["func_avgpool"] = F.avg_pool3d
            layers["interp_mode"] = 'trilinear'
        else:
            layers["layer_conv"] = nn.Conv2d
            layers["layer_convtrans"] = nn.ConvTranspose2d
            layers["layer_batchnorm"] = nn.BatchNorm2d
            layers["layer_drop"] = nn.Dropout2d
            layers["func_avgpool"] = F.avg_pool2d
            layers["interp_mode"] = 'bilinear'
        globals().update(layers)

        self.returnBlocks = returnBlocks
        self.do_down = downPath
        self.do_up = upPath

        self.padding = padding
        self.depth = depth
        self.dropout = layer_drop(p=droprate) 
        prev_channels = in_channels
        
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if self.do_down:
                self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                    padding, batch_norm))
            prev_channels = 2**(wf+i)
        self.latent_channels = prev_channels
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            if self.do_up:
                self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        if self.do_up:
            self.last = layer_conv(prev_channels, out_channels, kernel_size=1)

    def forward(self, x, blocks=()):
        if self.do_down:
            for i, down in enumerate(self.down_path):
                x = down(x)
                if i != len(self.down_path)-1:
                    blocks += (x,)
                    x = func_avgpool(x, 2)
            x = self.dropout(x)

        if self.do_up:
            for i, up in enumerate(self.up_path):
                x = up(x, blocks[-i-1])
            x = self.last(x)

        if self.returnBlocks and self.do_down:
            return x, blocks
        else:
            return x

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,64,64)).cuda()
    model = UNet().cuda()
    y = model(x)
    print(y.shape)
