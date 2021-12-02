import sys
import torch
from Engineering.math.freq_trans import fftNc, ifftNc
import torchkbnufft as tkbn
import numpy as np

class DataConsistency():
    def __init__(self, isRadial=False, metadict=None):
        self.isRadial = isRadial
        self.metadict = metadict
        # if isRadial:
        #     sys.exit("DataConsistency: Not working for Radial yet, due to raw kSpace troubles")

    # def cartesian_fastmri(self, out_ksp, full_ksp, under_ksp, mask): #torch.where doesn't work with complex
    #     if mask is None:
    #         mask = self.mask
    #     mask = mask.to(out_ksp.device)
    #     missing_mask = 1-mask
    #     missing_ksp = torch.where(missing_mask == 0, torch.Tensor([0]).to(out_ksp.device), out_ksp)
    #     if under_ksp is None:
    #         under_ksp = torch.where(mask == 0, torch.Tensor([0]).to(full_ksp.device), full_ksp)
    #     out_corrected_ksp = under_ksp + missing_ksp
    #     return out_corrected_ksp

    def cartesian_Ndmask(self, out_ksp, full_ksp, under_ksp, metadict):
        mask = metadict["mask"] if type(metadict) is dict else metadict
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.to(out_ksp.device)
        if len(full_ksp.shape) == 3: #TODO: do it nicely, its too strict now
            mask = mask.unsqueeze(-1)
        missing_mask = 1-mask
        missing_ksp = out_ksp * missing_mask
        if under_ksp is None:
            under_ksp = full_ksp * mask
        out_corrected_ksp = under_ksp + missing_ksp
        return out_corrected_ksp

    def radial(self, out_ksp, full_ksp, under_ksp, metadict):
        # om = torch.from_numpy(metadict['om'].transpose()).to(torch.float).to("cuda")
        # invom = torch.from_numpy(metadict['invom'].transpose()).to(torch.float).to("cuda")
        # fullom = torch.from_numpy(metadict['fullom'].transpose()).to(torch.float).to("cuda")
        # # dcf = torch.from_numpy(metadict['dcf'].squeeze())
        # dcfFullRes = torch.from_numpy(metadict['dcfFullRes'].squeeze()).to(torch.float).to("cuda")
        baseresolution = out_ksp.shape[0]*2
        Nd = (baseresolution, baseresolution)
        imsize = out_ksp.shape[:2]

        nufft_ob = tkbn.KbNufft(
            im_size=imsize,
            grid_size=Nd,
        ).to(torch.complex64).to("cuda")
        adjnufft_ob = tkbn.KbNufftAdjoint(
            im_size=imsize,
            grid_size=Nd,
        ).to(torch.complex64).to("cuda")

        # intrp_ob = tkbn.KbInterp(
        #     im_size=imsize,
        #     grid_size=Nd,
        # ).to(torch.complex64).to("cuda")

        out_img = ifftNc(data=out_ksp, dim=(0,1), norm="ortho").to("cuda")
        full_img = ifftNc(data=full_ksp, dim=(0,1), norm="ortho").to("cuda")

        out_img = torch.permute(out_img, dims=(2,0,1)).unsqueeze(1)
        full_img = torch.permute(full_img, dims=(2,0,1)).unsqueeze(1)

        # out_img = torch.permute(out_ksp, dims=(2,0,1)).unsqueeze(1).to("cuda")
        # full_img = torch.permute(full_ksp, dims=(2,0,1)).unsqueeze(1).to("cuda")

        spokelength = full_img.shape[-1] * 2
        grid_size = (spokelength, spokelength)
        nspokes = 512

        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
            
        ky = np.transpose(ky)
        kx = np.transpose(kx)

        fullom = torch.from_numpy(np.stack((ky.flatten(), kx.flatten()), axis=0)).to(torch.float).to("cuda")
        om = fullom[:,:30720]
        invom = fullom[:,30720:]
        dcfFullRes = tkbn.calc_density_compensation_function(ktraj=fullom, im_size=imsize).to("cuda")

        yUnder = nufft_ob(full_img, om, norm="ortho")
        yMissing = nufft_ob(out_img, invom, norm="ortho")
        # yUnder = intrp_ob(full_img, om)
        # yMissing = intrp_ob(out_img, invom)
        yCorrected = torch.concat((yUnder,yMissing), dim=-1)
        yCorrected = dcfFullRes * yCorrected
        out_corrected_img = adjnufft_ob(yCorrected, fullom, norm="ortho").squeeze()

        out_corrected_img = torch.abs(out_corrected_img)
        out_corrected_img = (out_corrected_img - out_corrected_img.min()) / (out_corrected_img.max() - out_corrected_img.min())

        out_corrected_img = torch.permute(out_corrected_img, dims=(1,2,0))

        out_corrected_ksp = fftNc(data=out_corrected_img, dim=(0,1), norm="ortho").cpu()
        return out_corrected_ksp

    def apply(self, out_ksp, full_ksp, under_ksp, metadict=None):
        if metadict is None:
            metadict = self.metadict
        if self.isRadial:
            return self.radial(out_ksp, full_ksp, under_ksp, metadict)
        else:
            return self.cartesian_Ndmask(out_ksp, full_ksp, under_ksp, metadict)