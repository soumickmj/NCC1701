#!/usr/bin/env python

import torch
import torch.nn as nn
from tricorder.math.transforms.fourier import fftNc_pyt, ifftNc_pyt
from tricorder.torch.transforms import Interpolator

from .KSPReconResNet import ResNet as KspResNet
from .ReconResNet import ResNet as ImResNet

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2021, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"


class DualSpaceResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="sigmoid", forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False, connect_mode="w_parallel", inner_norm_ksp=True):
        super(DualSpaceResNet, self).__init__()

        self.imnet = ImResNet(in_channels=in_channels, out_channels=out_channels, res_blocks=res_blocks, starting_nfeatures=starting_nfeatures, updown_blocks=updown_blocks, is_relu_leaky=is_relu_leaky, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                              is_replicatepad=is_replicatepad, out_act="relu" if "parallel" in connect_mode else "", forwardV=forwardV, upinterp_algo=upinterp_algo, post_interp_convtrans=post_interp_convtrans, is3D=is3D)

        self.kspnet = KspResNet(n_channels=in_channels, out_channels=out_channels, res_blocks=res_blocks, starting_nfeatures=starting_nfeatures, updown_blocks=0, is_relu_leaky=True, do_batchnorm=do_batchnorm, res_drop_prob=res_drop_prob,
                                is_replicatepad=is_replicatepad, out_act="relu", forwardV=forwardV, upinterp_algo='sinc', post_interp_convtrans=post_interp_convtrans, is3D=is3D,
                                img_out_mode=0 if "parallel" in connect_mode else 2, fourier_norm_4imgout="ortho", under_replace=False, under_mask=None, inner_norm_ksp=inner_norm_ksp)

        self.connect_mode = connect_mode
        if connect_mode == "w_parallel":
            self.parallel_kspweight = torch.nn.parameter.Parameter(torch.ones(1)*0.5)

        if out_act == "sigmoid":
            self.finalact = nn.Sigmoid()

    def forward(self, x):
        if "parallel" in self.connect_mode:
            x_imnet_ksp = fftNc_pyt(self.imnet(x))
            x_kspnet_ksp = self.kspnet(fftNc_pyt(x))
            if self.connect_mode == "w_parallel":
                x_ksp = (1-self.parallel_kspweight)*x_imnet_ksp + \
                    self.parallel_kspweight*x_kspnet_ksp
            else:
                x_ksp = (0.5*x_imnet_ksp) + (0.5*x_kspnet_ksp)
            out = torch.abs(ifftNc_pyt(x_ksp, norm="ortho"))
        elif self.connect_mode == "serial":
            x_kspnet = self.finalact(self.kspnet(fftNc_pyt(x)))
            out = self.imnet(x_kspnet)
        return self.finalact(out)
