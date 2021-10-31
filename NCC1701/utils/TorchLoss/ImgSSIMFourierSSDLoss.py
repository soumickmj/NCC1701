import torch
import pytorch_ssim
from Math.TorchFrequencyTransforms import ifft2cT

class ImgSSIMFourierSSDLoss(torch.nn.Module):
    def __init__(self):
        super(ImgSSIMFourierSSDLoss, self).__init__()

    def forward(self, input, target):
        inputI = ifft2cT(input).to(input.device)
        targetI = ifft2cT(target).to(input.device)
        ssimI = pytorch_ssim.SSIM()((inputI-inputI.min())/(inputI.max()-inputI.min()), (targetI-targetI.min())/(targetI.max()-targetI.min()))
        ssdF = torch.sum((input-target)**2)
        return ssdF - ssimI
