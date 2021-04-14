import torch
import numpy as np

class PearsonLoss(torch.nn.Module):
    def __init__(self, ver=1):
        super(PearsonLoss, self).__init__()

    def forward(self, img1, img2):
        if(ver == 1):
            x = output
            y = target
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        elif(ver == 2):
            x = output
            y = target
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            cost = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

        return cost