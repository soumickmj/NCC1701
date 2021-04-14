import torch
import pytorch_ssim
from Math.TorchFrequencyTransforms import ifht2cT

class ImgSSIMHartleySSDLoss(torch.nn.Module):
    def __init__(self):
        super(ImgSSIMHartleySSDLoss, self).__init__()

    def forward(self, input, target):
        inputI = ifht2cT(input).to(input.device)
        targetI = ifht2cT(target).to(input.device)
        ssimI = pytorch_ssim.SSIM()((inputI-inputI.min())/(inputI.max()-inputI.min()), (targetI-targetI.min())/(targetI.max()-targetI.min()))
        ssdF = torch.sum((input-target)**2)
        return ssdF - ssimI
