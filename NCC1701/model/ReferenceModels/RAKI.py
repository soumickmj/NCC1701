#!/usr/bin/env python

"""
In-house implimentation of RAKI
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class RAKI(nn.Module):
    def __init__(self, channels=8):
        super(RAKI, self).__init__()

        self.f1 = nn.Conv2d(channels*2, 32, kernel_size=(5,2), bias=False)
        self.f2 = nn.Conv2d(32, 8, kernel_size=(1,1), bias=False)
        self.f3 = nn.Conv2d(8, channels*2, kernel_size=(3,2), bias=False)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x


