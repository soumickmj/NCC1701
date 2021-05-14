import sys
import torch

class DataConsistency():
    def __init__(self, isRadial=False, mask=None):
        self.isRadial = isRadial
        self.mask = mask
        if isRadial:
            sys.exit("DataConsistency: Not working for Radial yet, due to raw kSpace troubles")

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

    def cartesian_Ndmask(self, out_ksp, full_ksp, under_ksp, mask):
        if mask is None:
            mask = self.mask
        mask = mask.to(out_ksp.device)
        missing_mask = 1-mask
        missing_ksp = out_ksp * missing_mask
        if under_ksp is None:
            under_ksp = full_ksp * mask
        out_corrected_ksp = under_ksp + missing_ksp
        return out_corrected_ksp

    def radial(self, out_ksp, full_ksp, under_ksp, mask):
        sys.exit("DataConsistency: Not working for Radial yet, due to raw kSpace troubles")

    def apply(self, out_ksp, full_ksp, under_ksp, mask):
        if self.isRadial:
            return self.radial(out_ksp, full_ksp, under_ksp, mask)
        else:
            return self.cartesian_Ndmask(out_ksp, full_ksp, under_ksp, mask)