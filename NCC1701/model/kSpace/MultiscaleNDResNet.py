#!/usr/bin/env python

"""
Implimention of Multi Scale nD ResNet for Inverse Problems, as well as for classifications.
Original work (for signal classification): https://github.com/geekfeiw/Multi-Scale-1D-ResNet

ResNet creator functions are based on torchvision ResNet model from PyTorch (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

The original work has been extensively modified.
This version is capable of having varible number of scales (original: fixed at [3,5,7]), levels for each scale (original: fixed at 3), starting_features (original: 64).
Addtionally, can be controlled whether the intial conv will downsample or not (original: will always downsample)
The major change has been the addtion of deconv, making it an auto-encoder suitable for reconstruction problems.
Can be used as simple auto-encoder without skip connections, as well as concat or residue for skip connection.
Interpolation or ConvTrans either can be used for upconv.
Can be used for classification as well (Then the upsample path will be ignored) [similar to the original work, but with additional functionalities]

Usage:-
1.  Standard ResNet models can be created using the creator functions. See below for the list of creator functions.
    If all the default params are desired, then call those creator functions without sending any parameters. 
    Params that can not be sent while using these creator functions are: useBottleneck, n_layers, layers, groups, width_per_group. As they are controlled by the standards.
    Any other prams of the MSResNet constructor can be sent, if their default values needs to be modified.
    Allowed params with these creator functions: BigD, in_channels, out_channel, n_scales, starting_nplanes, act, skipmode, interp4upconv,  downsamAtStart, dilation, n_fullScaleBlocks, preserveSizeInScales, doClassification
2. If custom ResNet model is to be created, then directly create object of the MSResNet class. Send values of the params for which default value isn't desired.

"""
import sys
import numpy as np
from itertools import chain
import torch.nn as nn
import torch
import torch.nn.functional as F

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Soumick Chatterjee & OvGU:ESF:MEMoRIAL & MedDigit"
__credits__ = ["Soumick Chatterjee", "Alessandro Sciarra", "Max Dünwwald"]

__license__ = "GPL"
__version__ = "0.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Haven't been tested yet"

#######Base Convolutions ConvTransponses###########
def _conv(in_planes, out_planes, kernel, stride=1, interpolation=None, groups=1, dilation=1):
    """convolution with padding"""
    #interpolation has no utility here. Just been added to have same function signature as upconv
    return pyt_conv(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=1, bias=False, groups=groups, dilation=dilation)

def _upconv(in_planes, out_planes, kernel, stride=1, interpolation=None, groups=1, dilation=1):
    """de-convolution with padding
    set interpolation to None if ConvTrans is to be used"""
    if interpolation is not None:
        return nn.Sequential(
            nn.Upsample(scale_factor=stride, mode=interpolation, align_corners=True),
            _conv(in_planes, out_planes, kernel, stride=1, groups=groups, dilation=dilation)
        )
    else:
        return pyt_convT(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=1, output_padding=0 if stride==1 else 1, bias=False, groups=groups, dilation=dilation)


#########Basic Block###########
class BasicBlock(nn.Module):
    expansion = 1 #Not in used, just been added to maintain consistency 

    def __init__(self, inplanes, planes, kernel, stride=1, downupsample=None, isup=False, act=nn.PReLU, interp4upconv=None, groups=1,
                 base_width=64, dilation=1, semifinalconv=False, drop_prob=0.0):
        #groups, base_width, dilation, semifinalconv: Not been used here. Just been added to have same function signature as Bottleneck
        super(BasicBlock, self).__init__()
        if isup:
            convlayer = _upconv
        else:
            convlayer = _conv
        self.isup = isup

        self.conv1 = convlayer(inplanes, planes, kernel, stride, interpolation=interp4upconv)
        self.bn1 = pyt_bn(planes)   
        self.do = drop(drop_prob)     
        self.relu1 = act(num_parameters=planes) if act == nn.PReLU else act()
        self.conv2 = convlayer(planes, planes, kernel, interpolation=interp4upconv)
        self.bn2 = pyt_bn(planes)
        self.relu2 = act(num_parameters=planes) if act == nn.PReLU else act()
        self.downupsample = downupsample
        self.stride = stride

    def forward(self, x):
        residual = x 

        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu1(self.do(out))

        out = self.conv2(out) 
        out = self.bn2(out)

        if self.downupsample is not None:
            residual = self.downupsample(x) 

        d = np.subtract(residual.shape , out.shape)[2:]
        if not np.any(d):
            out += residual
        elif d.min() > 0:
            d = np.abs(d)
            out = residual + F.pad(out, tuple(chain(*[[k//2, k-(k//2)] for k in d])))
        else:
            out = out[:,:,0:d[0],...]
            if len(d) > 1:
                out = out[:,:,:,0:d[1],...]
            if len(d) > 2:
                out = out[:,:,:,:,0:d[2],...]
            out = residual + out

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel, stride=1, downupsample=None, isup=False, act=nn.PReLU, interp4upconv=None, groups=1,
                 base_width=64, dilation=1, semifinalconv=False, drop_prob=0.0):
        super(Bottleneck, self).__init__()
        if isup:
            convlayer = _upconv
        else:
            convlayer = _conv
        self.isup = isup
        width = int(planes * (base_width / 64.)) * groups

        expansion_local = 1 if semifinalconv else self.expansion

        self.dilation = dilation
        self.groups = groups

        self.conv1 = convlayer(inplanes, width, 1, 1, interpolation=interp4upconv)
        self.bn1 = pyt_bn(width)  
        self.do1 = drop(drop_prob)  
        self.relu1 = act(num_parameters=width) if act == nn.PReLU else act()
        self.conv2 = convlayer(width, width, kernel, stride, interpolation=interp4upconv, groups=groups, dilation=dilation)
        self.bn2 = pyt_bn(width)  
        self.do2 = drop(drop_prob)  
        self.relu2 = act(num_parameters=width) if act == nn.PReLU else act()
        self.conv3 = convlayer(width, planes * expansion_local, 1, 1, interpolation=interp4upconv)
        self.bn3 = pyt_bn(planes * expansion_local)
        self.relu3 = act(num_parameters=planes * expansion_local) if act == nn.PReLU else act()
        self.downupsample = downupsample
        self.stride = stride

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(self.do1(out))

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(self.do2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downupsample is not None:
            residual = self.downupsample(x) 
            
        d = np.subtract(residual.shape , out.shape)[2:]
        if not np.any(d):
            out += residual
        elif d.min() > 0:
            d = np.abs(d)
            out = residual + F.pad(out, tuple(chain(*[[k//2, k-(k//2)] for k in d])))
        else:
            out = out[:,:,0:d[0],...]
            if len(d) > 1:
                out = out[:,:,:,0:d[1],...]
            if len(d) > 2:
                out = out[:,:,:,:,0:d[2],...]
            out = residual + out
        
        out = self.relu3(out) 

        return out


########Main ResNet############
class MSResNet(nn.Module):
    def __init__(self, BigD=1, in_channels=1, out_channel=1, n_layers=3, n_scales=3, layers=1, useBottleneck=True, starting_nplanes=64, act=nn.PReLU, skipmode=0, interp4upconv=None, 
                downsamAtStart=True, groups=1, width_per_group=64, dilation=1, drop_prob=0.5, n_fullScaleBlocks=3, preserveSizeInScales=False, doClassification=False):
        #BigD: The dimenstion of the data [1, 2 or 3] (as per original work, 1)
        #n_layers: how many layers each scales should have (as per original work, 3)
        #n_scales: how many scales this MSResNet should have (as per original work, 3)
        #layers: how many layers each level of a particular scale should have. Each element is for each scale. (as per original work, 1 1 1) 
        #        [Can be supplied either as a list or as an integer. If Integer is supplied, then same will be used for all the lev]
        #starting_nplanes: how many planes the first Conv should give as output, and the multi-scale Res blocks will have to start with (original: 64)
        #act: Activation function, refernce to pytorch activation functions or custom activation functions
        #skipmode: 0 = no skip connection (simple autoencoder), 1 = concat for skipconnection (UNet like), 2 = residual skip connection
        #interp4upconv: None = for using ConvTrans, OR else, put the string for interpolation mode
        #downsamAtStart: the intial conv will shirnk the input size to half, consequently the final conv will get the size back. Multi-scale Res blocks will work with half of input size
        #groups: grouped convolution (Number of blocked connections from input channels to output channels) (as per original work, not been used. By default, 1)
        #dilation: Spacing between kernel elements (as per original work, not been used. By default, 1)
        #drop_prob: Dropout probability, to be used within the Basic or Bottleneck Blocks
        #fullScaleBlocks: Apply 1 level of residual BasicBlocks. This param sets the no of blocks (layers) to be used. Set it to 0 if not desired. (as per original work, not been used. By default, 0)
        #preserveSizeInScales: Size won't decrease during inside Multiscale Res layers. i.e. Stride=1 (original work: False)
        #doClassification: If the model is to be used for classification, then set to True. If True, then upsample layers won't be created and after downsample layers, Liner FC will be used (original work: True)
        super(MSResNet, self).__init__()

        global pyt_conv, pyt_convT, pyt_bn, pyt_finalpool, drop
        if BigD==1:
            pyt_conv = nn.Conv1d
            pyt_convT = nn.ConvTranspose1d
            pyt_bn = nn.BatchNorm1d
            drop = nn.Dropout
            pyt_finalpool = F.adaptive_avg_pool1d
        elif BigD==2:
            pyt_conv = nn.Conv2d
            pyt_convT = nn.ConvTranspose2d
            pyt_bn = nn.BatchNorm2d
            drop = nn.Dropout2d
            pyt_finalpool = F.adaptive_avg_pool2d
        elif BigD==3: #Couldn't be tested because of lack of memory. In theory, should work.
            pyt_conv = nn.Conv3d
            pyt_convT = nn.ConvTranspose3d
            pyt_bn = nn.BatchNorm3d
            drop = nn.Dropout3d
            pyt_finalpool = F.adaptive_avg_pool3d
        else:
            sys.exit('Invalid Choise for Dim (BigD)')

        if type(layers) == int:
            layers = [layers] * n_layers

        assert len(layers) == n_layers, 'layers should have no of elements same as n_layers'

        self.act = act

        self.multiscale_inplanes = [starting_nplanes] * n_scales
        self.kernels = [3+(2*i) for i in range(n_scales)]

        self.n_layers = n_layers
        self.n_scales = n_scales
        self.skipmode = skipmode
        self.interp4upconv = interp4upconv

        self.groups = groups
        self.base_width = width_per_group
        self.dilation = dilation
        self.drop_prob = drop_prob

        self.doClassification = doClassification

        Block = Bottleneck if useBottleneck else BasicBlock

        self.conv1 = pyt_conv(in_channels, starting_nplanes, kernel_size=7, stride=2 if downsamAtStart else 1, padding=3,
                               bias=False)
        self.bn1 = pyt_bn(starting_nplanes)
        self.relu = act(num_parameters=starting_nplanes) if act == nn.PReLU else act()

        if n_fullScaleBlocks > 0:
            layer, _, _ = self._make_layers(BasicBlock, [n_fullScaleBlocks], n_layers=1, scale_index=-1, n_planes=[starting_nplanes], kernel=3, stride=1)
            self.fullScaleBlocks = nn.Sequential(*layer)
        else:
            self.fullScaleBlocks = None

        self.downlayers = nn.ModuleList()
        self.uplayers = nn.ModuleList()
        self.semifinalConvs = nn.ModuleList()
        for i in range(n_scales):
            downlayer, uplayer, semifinallayer = self._make_layers(Block, layers, n_layers=n_layers, scale_index=i, stride=1 if preserveSizeInScales else 2)
            self.downlayers.append(downlayer)
            self.uplayers.append(uplayer)
            self.semifinalConvs.append(semifinallayer)

        if doClassification:
            self.fc = nn.Linear(sum(self.multiscale_inplanes), out_channel)
        else:
            self.fc = pyt_convT(sum(self.multiscale_inplanes), out_channel, kernel_size=7, stride=2 if downsamAtStart else 1, padding=3, 
                output_padding=1 if downsamAtStart else 0, bias=False)

    def _make_layers(self, block, blocks, n_layers, scale_index, n_planes=None, kernel=None, stride=2):
        if n_planes is None:
            n_planes = [(2*i if i!=0 else 1)*self.multiscale_inplanes[scale_index] for i in range(n_layers)]
        if kernel is None:
            kernel = self.kernels[scale_index]

        #Create downsampling layer modules
        downlayer_modules = nn.ModuleList()
        for i in range(n_layers):
            planes = n_planes[i]
            n_block = blocks[i]
            downsample = None
            if stride != 1 or (n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]) != planes * block.expansion:
                downsample = nn.Sequential(
                    pyt_conv((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False),

                    pyt_bn(planes * block.expansion),
                )
            downlayers = []
            downlayers.append(block((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes, kernel, stride, downsample, isup=False, act=self.act, groups=self.groups,
                 base_width=self.base_width, dilation=self.dilation, drop_prob=self.drop_prob))
            if scale_index != -1:
                self.multiscale_inplanes[scale_index] = planes * block.expansion
            for i in range(1, n_block):
                downlayers.append(block((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes, kernel, isup=False, act=self.act, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation, drop_prob=self.drop_prob))
            
            downlayer_modules.append(nn.Sequential(*downlayers))

        #Create upsampling layer modules
        if n_layers > 1:
            n_planes.pop()
            n_planes = [n_planes[0]] + n_planes
        uplayer_modules = nn.ModuleList()
        semifinal_modules = nn.ModuleList()
        if not self.doClassification:
            for i in reversed(range(n_layers)):
                planes = n_planes[i]
                n_block = blocks[i]
                upsample = None
                if stride != 1 or (n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]) != planes * block.expansion:
                    upsample = nn.Sequential(
                        pyt_convT((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes * block.expansion,
                                    kernel_size=1, stride=stride, output_padding=0 if stride==1 else 1, bias=False),
                        pyt_bn(planes * block.expansion),
                    )                  
                uplayers = []
                uplayers.append(block((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes, kernel, stride, upsample, isup=True, act=self.act, interp4upconv=self.interp4upconv, 
                    groups=self.groups, base_width=self.base_width, dilation=self.dilation, drop_prob=self.drop_prob))
                if scale_index != -1:
                    self.multiscale_inplanes[scale_index] = planes * block.expansion
                for i in range(1, n_block):
                    uplayers.append(block((n_planes[0] if scale_index==-1 else self.multiscale_inplanes[scale_index]), planes, kernel, isup=True, act=self.act, interp4upconv=self.interp4upconv, 
                        groups=self.groups, base_width=self.base_width, dilation=self.dilation, drop_prob=self.drop_prob))

                uplayer_modules.append(nn.Sequential(*uplayers))

                #Conv after Skip Connections. DoubleConv when not using skip connections
                resi_connector = None
                semifinal_inplanes = planes*(2 if self.skipmode==1 else 1)* block.expansion
                if self.skipmode==1:
                    resi_connector = nn.Sequential(
                        pyt_conv(semifinal_inplanes, planes * block.expansion,
                                    kernel_size=1, stride=1, bias=False),
                        pyt_bn(planes * block.expansion),
                    ) 
                semifinal_modules.append(block(semifinal_inplanes, planes* block.expansion, kernel, downupsample=resi_connector, isup=False, act=self.act, 
                    groups=self.groups, base_width=self.base_width, dilation=self.dilation, semifinalconv=True, drop_prob=self.drop_prob))
            


        return downlayer_modules, uplayer_modules, semifinal_modules

    def forward(self, x0):
        x0 = self.conv1(x0) 
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        if self.fullScaleBlocks is not None:
            x0 = self.fullScaleBlocks(x0)

        multiscale_x = []
        for i in range(self.n_scales):
            multiscale_x.append([x0])
            for j in range(self.n_layers):
                x = self.downlayers[i][j](multiscale_x[i][-1])
                multiscale_x[i].append(x)
            if not self.doClassification:
                for j in range(self.n_layers):
                    x = multiscale_x[i].pop()
                    if self.skipmode==0:
                        x = self.uplayers[i][j](x)
                    elif self.skipmode==1:
                        x = torch.cat([multiscale_x[i].pop(), self.uplayers[i][j](x)], dim=1)
                    else:
                        x = multiscale_x[i].pop() + self.uplayers[i][j](x)
                    multiscale_x[i].append(self.semifinalConvs[i][j](x))

        mutiscale_finals = [x[-1] for x in multiscale_x]

        out = torch.cat(mutiscale_finals, dim=1) #stacked the features

        if self.doClassification:
            out = pyt_finalpool(out, 1).squeeze()
        
        out = self.fc(out)

        return out


def _msresnet(arch, pretrained, **kwargs):
    model = MSResNet(**kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
        sys.exit('Pretrained option not yet been implimented, hense, should be set to False')
    return model


##############ResNet creator functions########################
def msresnet18(pretrained=False, **kwargs):
    return _msresnet(arch='msresnet18', pretrained=pretrained, useBottleneck=False, n_layers=4, layers=[2, 2, 2, 2], 
                   **kwargs)


def msresnet34(pretrained=False, **kwargs):
    return _msresnet(arch='msresnet34', pretrained=pretrained, useBottleneck=False, n_layers=4, layers=[3, 4, 6, 3],
                   **kwargs)


def msresnet50(pretrained=False, **kwargs):
    return _msresnet(arch='msresnet50', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 6, 3],
                   **kwargs)


def msresnet101(pretrained=False, **kwargs):
    return _msresnet(arch='msresnet101', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 23, 3],
                   **kwargs)


def msresnet152(pretrained=False, **kwargs):
    return _msresnet(arch='msresnet152', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 8, 36, 3],
                   **kwargs)


def msresnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _msresnet(arch='msesnext50_32x4d', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 6, 3],
                   **kwargs)


def msresnext101_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _msresnet(arch='msresnext101_32x8d', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 23, 3],
                   **kwargs)


def msresnext152_32x8d(pretrained=False, **kwargs): 
    #addtional
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 12
    return _msresnet(arch='msresnext101_32x8d', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 8, 36, 3],
                   **kwargs)


def wide_msresnet50_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _msresnet(arch='wide_msresnet50_2', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 6, 3],
                   **kwargs)


def wide_msresnet101_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _msresnet(arch='wide_msresnet101_2', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 4, 23, 3],
                   **kwargs)


def wide_msresnet152_2(pretrained=False, **kwargs):
    #addtional
    kwargs['width_per_group'] = 64 * 2
    return _msresnet(arch='wide_msresnet152_2', pretrained=pretrained, useBottleneck=True, n_layers=4, layers=[3, 8, 36, 3],
                   **kwargs)


if __name__ == "__main__":
    # m = MSResNet()
    m = msresnet18()
    #m.cuda()
    t = torch.ones(10,1,32)#.cuda()
    m(t)
    print('eli')