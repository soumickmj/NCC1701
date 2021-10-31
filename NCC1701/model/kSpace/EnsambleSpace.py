import torch
from torch import nn
from torch.nn import functional as F

from model.kSpace.CUNet import CUNet
from model.ResNet.Resnet2Dv2b14 import ResNet

import utils.fastMRI.TorchDSTransforms as transformsFastMRI

class EnsambleSpace(nn.Module):
    def __init__(self, n_channels, k_net=CUNet, i_net=ResNet, isInputKSP=True,
                 k_drop_prob=0.2, k_down_factor=1.5, k_complex_latent=False, k_latent_conv=True, #kSpace Net Params
                 i_res_blocks=14, i_starting_n_features=64, i_updown_blocks=2, i_is_relu_leaky=True, i_final_out_sigmoid=True #ImageSpace Net Params
                 ):
        super(EnsambleSpace, self).__init__()
        
        self.k_net = k_net(n_channels, k_drop_prob, k_down_factor, k_complex_latent, k_latent_conv)
        self.i_net = i_net(n_channels//2, i_res_blocks, i_starting_n_features, i_updown_blocks, i_is_relu_leaky, i_final_out_sigmoid)

        self.isInputKSP = isInputKSP

    def forward(self, input): 
        #if not KSP, then convert input to ksp
        out_k, _ = self.k_net(input)

        out_i = self.i_net(transformsFastMRI.complex_abs(transformsFastMRI.ifft2(transformsFastMRI.channel_to_complex(input,True))))
        out_i_k = transformsFastMRI.complex_to_channel(transformsFastMRI.rfft2(out_i),True)

        out = out_k+out_i_k
        return out, None #TODO: HOTFIX