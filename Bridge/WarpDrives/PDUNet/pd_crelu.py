from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchcomplex.nn.functional as cF
from Engineering.math.freq_trans import fftNc_pyt, ifftNc_pyt
from Engineering.math.misc import NormUnorm
# from pytorch_radon import IRadon, Radon
# from pytorch_radon.filters import HannFilter
from torch import nn
from torchcomplex import nn as cnn

from .unet import UNet
from .cunet import CUNet


class PrePadDualConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[List[int], Tuple[int, int], int], complex=False, complex_weights=True):
        super().__init__()
        if complex:
            self.conv = cnn.Conv2d(in_channels, out_channels, kernel_size, complex_weights=complex_weights)
            with torch.no_grad():
                if type(self.conv.bias) == nn.ParameterList:
                    self.conv.bias[0].zero_()
                    self.conv.bias[1].zero_()
                else:
                    self.conv.bias.zero_()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            with torch.no_grad():
                self.conv.bias.zero_()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _angle_padding(self, inp):
        return F.pad(
            inp, (self.kernel_size[1]//2, self.kernel_size[1]//2, 0, 0), mode='circular')

    def _proj_padding(self, inp):
        return F.pad(
            inp, (0, 0, self.kernel_size[0]//2, self.kernel_size[0]//2), mode='replicate')

    def forward(self, x):
        return self.conv(self._proj_padding(self._angle_padding(x)))


class PrePadPrimalConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[List[int], Tuple[int, int], int], complex=False, complex_weights=True):
        super().__init__()
        if complex:
            self.conv = cnn.Conv2d(in_channels, out_channels, kernel_size, complex_weights=complex_weights)
            with torch.no_grad():
                if type(self.conv.bias) == nn.ParameterList:
                    self.conv.bias[0].zero_()
                    self.conv.bias[1].zero_()
                else:
                    self.conv.bias.zero_()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            with torch.no_grad():
                self.conv.bias.zero_()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def _lr_padding(self, inp: torch.Tensor):
        return F.pad(
            inp, (self.kernel_size[1]//2, self.kernel_size[1]//2, 0, 0), mode='replicate')

    def _tb_padding(self, inp: torch.Tensor):
        return F.pad(
            inp, (0, 0, self.kernel_size[0]//2, self.kernel_size[0]//2), mode='replicate')

    def forward(self, x: torch.Tensor):
        return self.conv(self._tb_padding(self._lr_padding(x)))


class DualBlock(nn.Module):
    def __init__(self, features: int, complex=False, complex_weights=True):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadDualConv2D(features+2, 32, 3, complex=complex, complex_weights=complex_weights),
            cnn.CReLU() if complex else nn.PReLU(32),
            PrePadDualConv2D(32, 32, 3, complex=complex, complex_weights=complex_weights),
            cnn.CReLU() if complex else nn.PReLU(32),
            PrePadDualConv2D(32, features, 3, complex=complex, complex_weights=complex_weights),
        )
        self.diff_weight = nn.Parameter(torch.ones(1, features, 1, 1, dtype=torch.cfloat if complex else torch.float32))

    def forward(self, h: torch.Tensor, f: torch.Tensor, g: torch.Tensor):
        B, _, H, W = h.shape
        block_input = torch.cat([h, torch.mean(f, dim=1, keepdim=True), g], 1)
        return h + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalBlock(nn.Module):
    def __init__(self, features: int, complex=False, complex_weights=True):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadPrimalConv2D(features+1, 32, 3, complex=complex, complex_weights=complex_weights),
            cnn.CReLU() if complex else nn.PReLU(32),
            PrePadPrimalConv2D(32, 32, 3, complex=complex, complex_weights=complex_weights),
            cnn.CReLU() if complex else nn.PReLU(32),
            PrePadPrimalConv2D(32, features, 3, complex=complex, complex_weights=complex_weights),
        )
        if complex:
            self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1, dtype=torch.cfloat) + torch.finfo(torch.cfloat).eps)
        else:
            self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, h: torch.Tensor, f: torch.Tensor):
        B, _, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        return f + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalUnetBlock(nn.Module):
    def __init__(self, features: int, complex=False, complex_weights=False):
        super().__init__()        
        if complex:
            self.layers = CUNet(features+1, features, wf=5, complex_weights=complex_weights)
            self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1, dtype=torch.cfloat) + torch.finfo(torch.cfloat).eps)
        else:
            self.layers = UNet(features+1, features, wf=5)
            self.diff_weight = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, h, f):
        B, _, H, W = f.shape
        block_input = torch.cat([f, torch.mean(h, dim=1, keepdim=True)], 1)
        return f + self.diff_weight.repeat(B, 1, H, W)*self.layers(block_input)


class PrimalDualNetwork(nn.Module):
    def __init__(self,
                 n_primary: int, n_dual: int, n_iterations: int,
                 in_size: Optional[int] = None,
                 theta: Optional[Union[List[float],
                                       np.ndarray, torch.Tensor]] = None,
                 use_original_block: bool = True,
                 use_original_init: bool = True,
                 use_complex_primal: bool = False,
                 complex_weights=True,
                 f_normtype: str = "factor",
                 f_normfactor: Optional[float] = None,
                 g_normtype: str = "zscore",
                 transform="Fourier",
                 return_abs = False,
                 output_stages = False):
        super().__init__()
        self.primal_blocks = nn.ModuleList(
            [
                (PrimalBlock if use_original_block else PrimalUnetBlock)(
                    n_primary,
                    complex=use_complex_primal and transform == "Fourier",
                    complex_weights=complex_weights,
                )
                for _ in range(n_iterations)
            ]
        )

        self.dual_blocks = nn.ModuleList(
            [
                DualBlock(
                    n_dual,
                    complex=transform == "Fourier",
                    complex_weights=complex_weights,
                )
                for _ in range(n_iterations)
            ]
        )

        self.in_size = in_size
        self.n_primary = n_primary
        self.n_dual = n_dual
        self.use_original_init = use_original_init
        self.use_complex_primal = use_complex_primal
        if f_normtype == "factor":
            self.f_norm = NormUnorm(
                type="factor", factor=f_normfactor if bool(f_normfactor) else 1.0)
        self.f_normtype = f_normtype
        self.g_normtype = g_normtype
        self.transform = transform
        self.return_abs = return_abs
        self.output_stages = output_stages
        if transform == "Radon":
            self.T = Radon(in_size, theta, circle=False, scikit=True)
            self.iT = IRadon(in_size, theta, circle=False,
                             use_filter=HannFilter(), scikit=True)
        elif transform == "Fourier":
            self.T = fftNc_pyt
            self.iT = ifftNc_pyt

    def get_primal_dual_diff_weights(self):
        return {
            'primal': [f.diff_weight.mean().item() for f in self.primal_blocks],
            'dual': [f.diff_weight.mean().item() for f in self.dual_blocks]
        }

    def forward(self, f0: torch.Tensor, g: torch.Tensor=None):
        if g is None:
            g = self.T(f0)

        g_norm = NormUnorm(g, type=self.g_normtype)
        f_norm = self.f_norm if self.f_normtype == "factor" else NormUnorm(
            f0, type=self.f_normtype)
        g = g_norm.normalise(g)
        B, _, P, A = g.shape
        h = torch.zeros(B, self.n_dual, P, A, device=g.device)
        if self.use_original_init:
            f = torch.zeros(B, self.n_primary, P, A, device=g.device)
        else:
            f = f_norm.normalise(f0.repeat(1, self.n_primary, 1, 1))

        stages = []
        for primary_block, dual_block in zip(self.primal_blocks, self.dual_blocks):
            h = dual_block(h, g_norm.normalise(
                self.T(f_norm.unnormalise(f))), g)
            _tmp_h = self.iT(g_norm.unnormalise(h))
            if not self.use_complex_primal and self.transform == "Fourier":
                _tmp_h = torch.abs(_tmp_h)
                if torch.is_complex(f):
                    f = torch.abs(f)
            f = primary_block(f_norm.normalise(_tmp_h), f)
            if self.output_stages:
                stages.append(torch.mean(f, dim=1, keepdim=True))

        out = torch.mean(f_norm.unnormalise(f), dim=1, keepdim=True)
        if self.use_complex_primal and self.transform == "Fourier" and self.return_abs:
            out = torch.abs(out)

        return (out, stages) if self.output_stages else out

    def custom_step(self, batch, slice_squeeze, loss_func):
        inpI, gtI = slice_squeeze(batch['inp']['data']), slice_squeeze(batch['gt']['data']) 
        inpK = slice_squeeze(batch['inp']['ksp'])
        outI = self(f0=inpI, g=inpK)
        loss = loss_func(outI, gtI.to(outI.dtype))
        return outI, loss

def test_timing_orig_vs_unet():
    import time
    net_orig = PrimalDualNetwork(
        256, np.arange(180)[::8], 5, 5, 10, use_original_block=True,
        use_original_init=True).cuda()
    sparse_sino = torch.zeros(4, 1, 363, 23, dtype=torch.float, device='cuda')
    under_reco = torch.zeros(4, 1, 256, 256, dtype=torch.float, device='cuda')
    # warmup
    for _ in range(5):
        net_orig(sparse_sino, under_reco)
    starttime = time.time()
    it = 20
    for _ in range(it):
        net_orig(sparse_sino, under_reco)
    print(f'original: {(time.time() - starttime)/it}')

    del net_orig
    net_unet = PrimalDualNetwork(
        256, np.arange(180)[::8], 4, 5, 2, use_original_block=False,
        use_original_init=False).cuda()
    # warmup
    for _ in range(5):
        net_unet(sparse_sino, under_reco)
    starttime = time.time()
    for _ in range(it):
        net_unet(sparse_sino, under_reco)
    print(f'unet: {(time.time() - starttime)/it}')


if __name__ == '__main__':
    test_timing_orig_vs_unet()
