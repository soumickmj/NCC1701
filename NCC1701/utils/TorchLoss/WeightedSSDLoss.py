import math
import numpy as np
import scipy.stats as stats
import numpy.matlib as mlib
import torch

class WeightedSSDLoss(torch.nn.Module):
    def __init__(self, shape=(256,256)):
        super(WeightedSSDLoss, self).__init__()
        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, shape[0])
        pdf = stats.norm.pdf(x, mu, sigma)
        self.weight = torch.from_numpy(mlib.repmat(pdf, shape[1], 1)).float()

    def forward(self, input, target):
        input = input * self.weight.to(input.device)
        target = target * self.weight.to(target.device)
        return torch.sum((input-target)**2)
