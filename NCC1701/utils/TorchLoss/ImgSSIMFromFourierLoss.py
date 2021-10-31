#import torch
#import pytorch_ssim
#from Math.TorchFrequencyTransforms import ifft2cT
#from Math.FrequencyTransforms import ifft2c

#class ImgSSIMFromFourierLoss(torch.nn.Module):
#    def __init__(self):
#        super(ImgSSIMFromFourierLoss, self).__init__()

#    def forward(self, input, target):
#        inputX = input.detach().cpu().numpy()
#        inputXX = inputX.transpose((1, 2, 0))
#        inputXXX = ifft2c(inputXX)
#        targetX = target.detach().cpu().numpy()
#        targetXX = targetX.transpose((1, 2, 0))
#        targetXXX = ifft2c(targetXX)
#        return -pytorch_ssim.SSIM()(input, target)


import torch
import pytorch_ssim
#from Math.TorchFrequencyTransforms import ifft2cT
from Math.FrequencyTransforms import irfft2c as ifft2cT

class ImgSSIMFromFourierLoss(torch.nn.Module):
    def __init__(self):
        super(ImgSSIMFromFourierLoss, self).__init__()

    def forward(self, input, target):
        input = ifft2cT(input)
        target = ifft2cT(target)
        return pytorch_ssim.SSIM()((input-input.min())/(input.max()-input.min()), (target-target.min())/(target.max()-target.min()))
