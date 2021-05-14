from torch import nn
from torchcomplex import nn as cnn


class BasicBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob, if_norm=False, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if if_norm:
            self.layers = nn.Sequential(
                cnn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
                cnn.InstanceNorm2d(out_chans),
                cnn.PReLU(out_chans),
                cnn.Dropout2d(drop_prob),
                cnn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding=padding),
                cnn.InstanceNorm2d(out_chans),
                cnn.PReLU(out_chans),
                cnn.Dropout2d(drop_prob)
            )
        else:
            self.layers = nn.Sequential(
                cnn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
                cnn.AdaptiveCmodReLU(), #TODO should have n number of params, same as out_chans
                cnn.Dropout2d(drop_prob),
                cnn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding=padding),
                cnn.AdaptiveCmodReLU(),
                cnn.Dropout2d(drop_prob)
            )


    def forward(self, input):
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ResNet(nn.Module): #TODO: change class name, just for help of coding main now
    def __init__(self, n_channels, drop_prob=0.0, down_factor=1.5, latent_conv=True, return_latent=False):
        super(ResNet, self).__init__()
        self.return_latent=return_latent
        
        #c16 df2: 8,4,2,1 (7,5,3,1)
        #c16 df4: 4,1 (3,1)
        #c32 df4: 8,2,1 (5,3,1)
        #c32 df2: 16,8,4,2,1 (9,7,5,3,1)

        latent_size = 1

        channels = [n_channels]
        while channels[-1] != latent_size:
            next_channel = int(channels[-1] // down_factor)
            if next_channel < latent_size:
                next_channel = latent_size
            channels.append(next_channel)

        nDownBlocks = len(channels) - 1

        kernels = list(range((1+((nDownBlocks-1)*2)),0,-2))
        paddings = list(range(nDownBlocks-1,-1,-1))

        self.latent_conv = latent_conv

        self.down_layers = nn.ModuleList()
        for i in range(nDownBlocks): #Down blocks
            self.down_layers += [BasicBlock(channels[i], channels[i+1], drop_prob, False, kernels[i], paddings[i])]

        if latent_conv:
            self.latent_conv = BasicBlock(channels[-1], channels[-1], drop_prob, False, 1, 0)
        else:
            self.latent_conv = None

        self.up_layers = nn.ModuleList()
        for i in range(nDownBlocks-1, -1, -1): #Up blocks
            self.up_layers += [BasicBlock(channels[i+1], channels[i], drop_prob, False, kernels[i], paddings[i])]

    def forward(self, input): #if input channel is 16
        stack = []
        output = input

        # Apply down blocks
        for layer in self.down_layers:
            output = layer(output)
            stack.append(output)

        if self.latent_conv is not None:
            stack[-1] = self.latent_conv(stack[-1])
        latent = stack[-1]

        # Apply up blocks
        for layer in self.up_layers:
            output = output + stack.pop()
            output = layer(output)

        if self.return_latent:
            return output + input, latent
        else:
            return output + input