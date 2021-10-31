import math
import torch
import torch.nn as nn
import torch.nn.functional as f

class AttentionModule_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, ResidualBlock):
        super(AttentionModule_Basic, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        size = x.shape[2:]
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_softmax1_up = f.interpolate(out_softmax1, size=size, mode='bilinear', align_corners=False)
        out_interp1 = out_softmax1_up + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        return self.last_blocks(out)


class AttentionModule_Adv(nn.Module):
    def __init__(self, in_channels, out_channels, ResidualBlock, n_stage, current_stage, drop_prob=0.0):
        super(AttentionModule_Adv, self).__init__()

        #Actual stages can be upto 3, as per the paper
        stages_per_actualstage = math.ceil(n_stage/3)
        self.actual_stage = current_stage//stages_per_actualstage

        #Initial Blocks, present in all actual stages
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
         )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
            ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        )

        if self.actual_stage < 2: #For 1st and 2nd Stage
            self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.actual_stage == 1:
            self.softmax2_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            )
            self.softmax3_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
        elif self.actual_stage == 0:
            self.softmax2_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.softmax3_blocks = nn.Sequential(
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob),
                ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            )
            self.softmax4_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)
            self.softmax5_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)

        #Final Blocks, present in all actual stages
        self.softmaxN_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels, drop_prob=drop_prob)

    def forward(self, x):
        size = x.shape[2:]
        
        #Initial Blocks, present in all actual stages
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        if self.actual_stage == 2: #3rd Stage
            out_softmaxNm1 = out_softmax1
        else:
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            if self.actual_stage == 1: #2nd Stage                            
                out_softmax2_up = f.interpolate(out_softmax2, size=[x // 2 for x in size], mode='bilinear', align_corners=False)
                out_interp2 = out_softmax2_up + out_softmax1
                out = out_interp2 + out_skip1_connection
                out_softmaxNm1 = self.softmax3_blocks(out)
            elif self.actual_stage == 0: #1st Stage
                out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
                out_mpool3 = self.mpool3(out_softmax2)
                out_softmax3 = self.softmax3_blocks(out_mpool3)
                out_softmax3_up = f.interpolate(out_softmax3, size=[x // 4 for x in size], mode='bilinear', align_corners=False)
                out_interp3 = out_softmax3_up + out_softmax2
                out = out_interp3 + out_skip2_connection
                out_softmax4 = self.softmax4_blocks(out)
                out_softmax4_up = f.interpolate(out_softmax4, size=[x // 2 for x in size], mode='bilinear', align_corners=False)
                out_interp2 = out_softmax4_up + out_softmax1
                out = out_interp2 + out_skip1_connection
                out_softmaxNm1 = self.softmax5_blocks(out)

        #Final Blocks, present in all actual stages
        out_softmaxNm1_up = f.interpolate(out_softmaxNm1, size=size, mode='bilinear', align_corners=False)
        out_interpN = out_softmaxNm1_up + out_trunk
        out_softmaxN = self.softmaxN_blocks(out_interpN)
        out = (1 + out_softmaxN) * out_trunk
        return self.last_blocks(out)

#from model.Attention.ResNet.ResNetClassifier import ResidualBottleneckBlock
#m=AttentionModule_Adv(32,32,ResidualBottleneckBlock,5,2)
#z=torch.zeros(5,32,64,64)
#o=m(z)
#print('test')
