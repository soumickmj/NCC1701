import torch
import torch.nn as nn

class ComplexBatchNormal(nn.Module):
    def __init__(self, C, H, W, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.epsilon = 1e-5
        self.running_mean_real = None
        self.running_mean_imag = None
        self.Vrr = None
        self.Vri = None
        self.Vii = None

    def forward(self, x, train=True):
        B, C, H, W = x.size()
        real = x.real
        imaginary = x.imag
        if train:
            mu_real = torch.mean(real, dim=0)
            mu_imag = torch.mean(imaginary, dim=0)

            broadcast_mu_real = mu_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = mu_imag.repeat(B, 1, 1, 1)

            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag

            Vrr = torch.mean(real_centred * real_centred, 0) + self.epsilon
            Vii = torch.mean(imag_centred * imag_centred, 0) + self.epsilon
            Vri = torch.mean(real_centred * imag_centred, 0)
            if self.Vrr is None:
                self.running_mean_real = mu_real
                self.running_mean_imag = mu_imag
                self.Vrr = Vrr  # C,H,W
                self.Vri = Vri
                self.Vii = Vii
            else:
                # momentum
                self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, B)
        else:
            broadcast_mu_real = self.running_mean_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = self.running_mean_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            return self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, B)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, B):
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)

        Wrr = ((Vii + s) * inverse_st).repeat(B, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).repeat(B, 1, 1, 1)
        Wri = (-Vri * inverse_st).repeat(B, 1, 1, 1)

        n_real = Wrr * real_centred + Wri * imag_centred
        n_imag = Wii * imag_centred + Wri * real_centred

        broadcast_gamma_rr = self.gamma_rr.repeat(B, 1, 1, 1)
        broadcast_gamma_ri = self.gamma_ri.repeat(B, 1, 1, 1)
        broadcast_gamma_ii = self.gamma_ii.repeat(B, 1, 1, 1)
        broadcast_beta = self.beta.repeat(B, 1, 1, 1)

        bn_real = broadcast_gamma_rr * n_real + broadcast_gamma_ri * n_imag + broadcast_beta
        bn_imag = broadcast_gamma_ri * n_real + broadcast_gamma_ii * n_imag + broadcast_beta
        return bn_real + 1j*bn_imag