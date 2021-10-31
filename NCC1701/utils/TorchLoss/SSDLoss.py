import torch

class SSDLoss(torch.nn.Module):
    def __init__(self):
        super(SSDLoss, self).__init__()

    def forward(self, input, target):
        return torch.sum((input-target)**2)
