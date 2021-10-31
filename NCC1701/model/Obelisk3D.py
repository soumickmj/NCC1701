#!/usr/bin/env python

"""
Implimentation of https://openreview.net/forum?id=BkZu9wooz
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


#Hybrid OBELISK CNN model that contains two obelisk layers combined with traditional CNNs
#the layers have 512 and 128 trainable offsets and 230k trainable weights in total
class HybridObelisk3D(nn.Module):
    def __init__(self,out_channels,full_res):
        super(HybridObelisk3D, self).__init__()
        self.out_channels = out_channels
        D_in1 = full_res[0]; H_in1 = full_res[1]; W_in1 = full_res[2];
        D_in2 = (D_in1+1)//2; H_in2 = (H_in1+1)//2; W_in2 = (W_in1+1)//2; #half resolution
        self.half_res = torch.Tensor([D_in2,H_in2,W_in2]).long(); half_res = self.half_res
        D_in4 = (D_in2+1)//2; H_in4 = (H_in2+1)//2; W_in4 = (W_in2+1)//2; #quarter resolution
        self.quarter_res = torch.Tensor([D_in4,H_in4,W_in4]).long(); quarter_res = self.quarter_res
        D_in8 = (D_in4+1)//2; H_in8 = (H_in4+1)//2; W_in8 = (W_in4+1)//2; #eighth resolution
        self.eighth_res = torch.Tensor([D_in8,H_in8,W_in8]).long(); eighth_res = self.eighth_res

        #U-Net Encoder
        self.conv0 = nn.Conv3d(1, 4, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        
        # Obelisk Encoder (for simplicity using regular sampling grid)
        # the first obelisk layer has 128 the second 512 trainable offsets
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,quarter_res[0],quarter_res[1],quarter_res[2]))).view(1,1,-1,1,3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,eighth_res[0],eighth_res[1],eighth_res[2]))).view(1,1,-1,1,3).detach()
        self.sample_grid2.requires_grad = False
        
        self.offset1 = nn.Parameter(torch.randn(1,128,1,1,3)*0.05)
        self.linear1a = nn.Conv3d(4*128,128,1,groups=4,bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128,32,1,bias=False)
        self.batch1b = nn.BatchNorm3d(128+32)
        self.linear1c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch1c = nn.BatchNorm3d(128+64)
        self.linear1d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch1d = nn.BatchNorm3d(128+96)
        self.linear1e = nn.Conv3d(128+96,32,1,bias=False)
        
        self.offset2 = nn.Parameter(torch.randn(1,512,1,1,3)*0.05)
        self.linear2a = nn.Conv3d(512,128,1,groups=4,bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128,32,1,bias=False)
        self.batch2b = nn.BatchNorm3d(128+32)
        self.linear2c = nn.Conv3d(128+32,32,1,bias=False)
        self.batch2c = nn.BatchNorm3d(128+64)
        self.linear2d = nn.Conv3d(128+64,32,1,bias=False)
        self.batch2d = nn.BatchNorm3d(128+96)
        self.linear2e = nn.Conv3d(128+96,32,1,bias=False)
        
        #U-Net Decoder 
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64+16, 32, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(32)
        self.conv8 = nn.Conv3d(32, out_channels, 1)
        
    def forward(self, inputImg):
    
        B,C,D,H,W = inputImg.size()
        device = inputImg.device
        leakage = 0.05 #leaky ReLU used for conventional CNNs
        
        #unet-encoder
        x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)),leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)),leakage)
        
        #in this model two obelisk layers with fewer spatial offsets are used
        #obelisk layer 1
        x_o1 = F.grid_sample(x1, (self.sample_grid1.to(device).repeat(B,1,1,1,1) + self.offset1)).view(B,-1,self.quarter_res[0],self.quarter_res[1],self.quarter_res[2])
        #1x1 kernel dense-net
        x_o1 = F.relu(self.linear1a(x_o1))
        x_o1a = torch.cat((x_o1,F.relu(self.linear1b(self.batch1a(x_o1)))),dim=1)
        x_o1b = torch.cat((x_o1a,F.relu(self.linear1c(self.batch1b(x_o1a)))),dim=1)
        x_o1c = torch.cat((x_o1b,F.relu(self.linear1d(self.batch1c(x_o1b)))),dim=1)
        x_o1d = F.relu(self.linear1e(self.batch1d(x_o1c)))
        x_o1 = F.interpolate(x_o1d, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear', align_corners=False)
        
        #obelisk layer 2
        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B,1,1,1,1) + self.offset2)).view(B,-1,self.eighth_res[0],self.eighth_res[1],self.eighth_res[2])
        x_o2 = F.relu(self.linear2a(x_o2))
        #1x1 kernel dense-net
        x_o2a = torch.cat((x_o2,F.relu(self.linear2b(self.batch2a(x_o2)))),dim=1)
        x_o2b = torch.cat((x_o2a,F.relu(self.linear2c(self.batch2b(x_o2a)))),dim=1)
        x_o2c = torch.cat((x_o2b,F.relu(self.linear2d(self.batch2c(x_o2b)))),dim=1)
        x_o2d = F.relu(self.linear2e(self.batch2d(x_o2c)))
        x_o2 = F.interpolate(x_o2d, size=[self.quarter_res[0],self.quarter_res[1],self.quarter_res[2]], mode='trilinear', align_corners=False)

        #unet-decoder
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x_o2),1))),leakage)
        x = F.interpolate(x, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x_o1,x2),1))),leakage)
        x = F.interpolate(self.conv8(x), size=[D,H,W], mode='trilinear', align_corners=False)
        
        return x

#original OBELISK model as described in MIDL2018 paper
#contains around 130k trainable parameters and 1024 binary offsets
#most simple Obelisk-Net with one deformable convolution followed by 1x1 Dense-Net
class Obelisk3D(nn.Module):
    def __init__(self,out_channels,full_res):
        super(Obelisk3D, self).__init__()
        self.out_channels = out_channels
        self.full_res = full_res
        D_in1 = full_res[0]; H_in1 = full_res[1]; W_in1 = full_res[2];
        D_in2 = (D_in1+1)//2; H_in2 = (H_in1+1)//2; W_in2 = (W_in1+1)//2; #half resolution
        self.half_res = torch.Tensor([D_in2,H_in2,W_in2]).long(); half_res = self.half_res
        D_in4 = (D_in2+1)//2; H_in4 = (H_in2+1)//2; W_in4 = (W_in2+1)//2; #quarter resolution
        self.quarter_res = torch.Tensor([D_in4,H_in4,W_in4]).long(); quarter_res = self.quarter_res
        
        #Obelisk Layer
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        
        self.sample_grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0),torch.Size((1,1,quarter_res[0],quarter_res[1],quarter_res[2])))
        self.sample_grid1.requires_grad = False
        
        #in this model (binary-variant) two spatial offsets are paired 
        self.offset1 = nn.Parameter(torch.randn(1,1024,1,2,3)*0.05)
        
        #Dense-Net with 1x1x1 kernels
        self.LIN1 = nn.Conv3d(1024, 256, 1, bias=False, groups=4) #grouped convolutions
        self.BN1 = nn.BatchNorm3d(256)
        self.LIN2 = nn.Conv3d(256, 128, 1, bias=False)
        self.BN2 = nn.BatchNorm3d(128)
        
        self.LIN3a = nn.Conv3d(128, 32, 1,bias=False)
        self.BN3a = nn.BatchNorm3d(128+32)
        self.LIN3b = nn.Conv3d(128+32, 32, 1,bias=False)
        self.BN3b = nn.BatchNorm3d(128+64)
        self.LIN3c = nn.Conv3d(128+64, 32, 1,bias=False)
        self.BN3c = nn.BatchNorm3d(128+96)
        self.LIN3d = nn.Conv3d(128+96, 32, 1,bias=False)
        self.BN3d = nn.BatchNorm3d(256)
        
        self.LIN4 = nn.Conv3d(256, out_channels,1)

        
    def forward(self, inputImg, sample_grid=None):
    
        B,C,D,H,W = inputImg.size()
        if(sample_grid is None):
            sample_grid = self.sample_grid1
        sample_grid = sample_grid.to(inputImg.device)    
        #pre-smooth image (has to be done in advance for original models )
        #x00 = F.avg_pool3d(inputImg,3,padding=1,stride=1)
        
        _,D_grid,H_grid,W_grid,_ = sample_grid.size()
        input = F.grid_sample(inputImg, (sample_grid.view(1,1,-1,1,3).repeat(B,1,1,1,1) + self.offset1[:,:,:,0:1,:])).view(B,-1,D_grid,H_grid,W_grid)-\
        F.grid_sample(inputImg, (sample_grid.view(1,1,-1,1,3).repeat(B,1,1,1,1) + self.offset1[:,:,:,1:2,:])).view(B,-1,D_grid,H_grid,W_grid)
        
        x1 = F.relu(self.BN1(self.LIN1(input)))
        x2 = self.BN2(self.LIN2(x1))
        
        x3a = torch.cat((x2,F.relu(self.LIN3a(x2))),dim=1)
        x3b = torch.cat((x3a,F.relu(self.LIN3b(self.BN3a(x3a)))),dim=1)
        x3c = torch.cat((x3b,F.relu(self.LIN3c(self.BN3b(x3b)))),dim=1)
        x3d = torch.cat((x3c,F.relu(self.LIN3d(self.BN3c(x3c)))),dim=1)

        x4 = self.LIN4(self.BN3d(x3d))
        #return half-resolution segmentation/prediction 
        return F.interpolate(x4, size=[self.half_res[0],self.half_res[1],self.half_res[2]], mode='trilinear',align_corners=False)

if __name__=='__main__':
    from torchsummary import summary
    full_res = torch.Tensor([16,256,256]).long()
    model = HybridObelisk3D(1, full_res)#.cuda()
    summary(model, (1,16,256,256), device='cpu')