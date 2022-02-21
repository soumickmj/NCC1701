#!/usr/bin/env python

import torch
import torch.nn as nn
from tricorder.math.transforms.fourier import fftNc_pyt, ifftNc_pyt
from tricorder.torch.transforms import Interpolator

from Engineering.utilities import CustomInitialiseWeights

from .CentreKSPMoCoNet import CentreKSPMoCoNet
from .ReconResNet import ResNet as ImResNet

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2021, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class MoCoReCoNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="sigmoid", forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False, complex_moconet=False, inner_norm_ksp=True):
        super(MoCoReCoNet, self).__init__()

        self.reconet = ImResNet(in_channels=in_channels, out_channels=out_channels, res_blocks=res_blocks, starting_nfeatures=starting_nfeatures, updown_blocks=updown_blocks, is_relu_leaky=is_relu_leaky, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                              is_replicatepad=is_replicatepad, out_act=out_act, forwardV=forwardV, upinterp_algo=upinterp_algo, post_interp_convtrans=post_interp_convtrans, is3D=is3D)

        self.moconet = CentreKSPMoCoNet(in_channels=in_channels, out_channels=out_channels, res_blocks=56, starting_nfeatures=starting_nfeatures, updown_blocks=updown_blocks, is_relu_leaky=True, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                                        is_replicatepad=is_replicatepad, out_act="tanh", forwardV=forwardV, upinterp_algo=upinterp_algo, post_interp_convtrans=post_interp_convtrans, is3D=is3D,
                                        inner_norm_ksp=inner_norm_ksp, complex_moconet=complex_moconet)               
            
        self.apply(CustomInitialiseWeights)

    def forward(self, inp, gt=None, loss_func=None):
        mocoreturn = self.moconet(inp, gt, loss_func)
        if len(mocoreturn) == 2:
            mocoim_out, loss_moco = mocoreturn
        else:
            mocoim_out = mocoreturn
        recoim_out = self.reconet(mocoim_out)

        if gt is None or loss_func is None:
            return recoim_out

        loss_reco = loss_func(recoim_out, gt.to(recoim_out.dtype))

        return recoim_out, loss_moco+loss_reco

    def custom_step(self, batch, slice_squeeze, loss_func):
        inp, gt = slice_squeeze(batch['inp']['data']), slice_squeeze(batch['gt']['data'])

        mocoim_out, loss_moco = self.moconet(inp, gt, loss_func)

        recoim_out = self.reconet(mocoim_out)
        loss_reco = loss_func(recoim_out, gt.to(recoim_out.dtype))

        return recoim_out, loss_moco+loss_reco
