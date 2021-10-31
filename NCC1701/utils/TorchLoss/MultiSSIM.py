import torch.nn as nn
import pytorch_ssim
from utils.TorchLoss import SSIM3D

class MultiSSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True, is3D = False, reduction='mean'):
        super(MultiSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.is3D = is3D
        if is3D:            
            self.ssim_calculator = SSIM3D.SSIM3D(self.window_size, self.size_average) 
        else:
            self.ssim_calculator = pytorch_ssim.SSIM(self.window_size, self.size_average)  
        self.reduction = reduction


    def forward(self, img1, img2):
        count = 0
        for c in range(img1.shape[1]):
            if len(img1.shape) == 5 and not self.is3D:
                for d in range(img1.shape[2]):
                    ssim = self.ssim_calculator(img1[:,c,d,...].unsqueeze(1), img2[:,c,d,...].unsqueeze(1))
                    if 'ssims' in locals():
                        ssims += ssim
                    else:
                        ssims = ssim
                    count += 1
            else:
                ssim = self.ssim_calculator(img1[:,c,...].unsqueeze(1), img2[:,c,...].unsqueeze(1))
                if 'ssims' in locals():
                    ssims += ssim
                else:
                    ssims = ssim
                count += 1
        if self.reduction == 'sum':
            return ssims
        elif self.reduction == 'mean':
            return ssims/count