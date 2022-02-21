#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from tricorder.math.transforms.fourier import fftNc_pyt, ifftNc_pyt
from tricorder.torch.transforms import Interpolator

from Engineering.utilities import CustomInitialiseWeights

from .KSPReconResNet import ResNet as KspResNet
from .ReconResNet import ResNet as ImResNet

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2021, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class CentreKSPMoCoNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=56, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="tanh", forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False, complex_moconet=False, inner_norm_ksp=True):
        super(CentreKSPMoCoNet, self).__init__()

        self.complex_moconet = complex_moconet
        if complex_moconet:
            self.net = KspResNet(n_channels=in_channels, out_channels=out_channels, res_blocks=res_blocks, starting_nfeatures=starting_nfeatures, updown_blocks=updown_blocks, is_relu_leaky=is_relu_leaky, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                                is_replicatepad=is_replicatepad, out_act=out_act, forwardV=forwardV, upinterp_algo=upinterp_algo, post_interp_convtrans=post_interp_convtrans, is3D=is3D,
                                img_out_mode=0, fourier_norm_4imgout="ortho", under_replace=False, under_mask=None, inner_norm_ksp=inner_norm_ksp)
        else:
            self.net = ImResNet(in_channels=in_channels*2, out_channels=out_channels*2, res_blocks=res_blocks, starting_nfeatures=starting_nfeatures, updown_blocks=updown_blocks, is_relu_leaky=is_relu_leaky, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                                is_replicatepad=is_replicatepad, out_act=out_act, forwardV=forwardV, upinterp_algo=upinterp_algo, post_interp_convtrans=post_interp_convtrans, is3D=is3D,
                                )     
            
        self.apply(CustomInitialiseWeights)

    def forward(self, inp, gt=None, loss_func=None):
        ksp_inp = fftNc_pyt(inp, norm="ortho")
        assert inp.shape[-2] == 256 and inp.shape[-1] == 256, "Currently the centre crop is hardcoded for 256x256 images"
        ksp_inp = ksp_inp/torch.abs(ksp_inp).max()
        ksp_inp = ksp_inp[...,96:160,96:160]

        if not self.complex_moconet:
            ksp_inp = torch.cat([ksp_inp.real,ksp_inp.imag], dim=1)
        ksp_out = self.net(ksp_inp)
        if not self.complex_moconet:
            ksp_out = torch.view_as_complex(torch.stack(torch.split(ksp_out, ksp_out.shape[1]//2, dim=1), dim=-1))

        ksp_out = F.pad(ksp_out, (96,96,96,96))
        mocoim_out = ifftNc_pyt(ksp_out, norm="ortho")
        mocoim_out = torch.abs(mocoim_out/torch.abs(mocoim_out).max())     

        if gt is None or loss_func is None:
            return mocoim_out
        
        ksp_gt = fftNc_pyt(gt, norm="ortho")
        ksp_gt = ksp_gt/torch.abs(ksp_inp).max()
        ksp_gt = ksp_gt[...,96:160,96:160]
        ksp_gt = F.pad(ksp_gt, (96,96,96,96))
        mocoim_gt = ifftNc_pyt(ksp_gt, norm="ortho")
        mocoim_gt = torch.abs(mocoim_gt/torch.abs(mocoim_gt).max())

        loss_moco = loss_func(mocoim_out, mocoim_gt.to(mocoim_out.dtype))
        return mocoim_out, loss_moco

    def custom_step(self, batch, slice_squeeze, loss_func):
        inp, gt = slice_squeeze(batch['inp']['data']), slice_squeeze(batch['gt']['data'])
        return self(inp, gt, loss_func)