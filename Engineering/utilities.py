import os
from collections import OrderedDict
from statistics import median

import nibabel as nib
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import torch
import torchcomplex
import torchio as tio
import torchvision.utils as vutils
from async_timeout import sys
from sewar.full_ref import ssim as SSIM2DCalc
from sewar.full_ref import uqi as UQICalc
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)

from Engineering.data_consistency import DataConsistency
from Engineering.math.freq_trans import fftNc, ifftNc
from Engineering.transforms.tio.transforms import getDataSpaceTransforms


def sitkShow(data, slice_last=True):
    if issubclass(type(data), tio.Image):
        data = data['data']
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = data.squeeze()
    if slice_last and len(data.shape) == 3:
        data = np.transpose(data)
    img = sitk.GetImageFromArray(data)
    sitk.Show(img)


def getSSIM(gt, out, gt_flag=None, data_range=1):
    if gt_flag is None:  # all of the samples have GTs
        gt_flag = [True]*gt.shape[0]

    vals = []
    for i in range(gt.shape[0]):
        if not gt_flag[i]:
            continue
        for j in range(gt.shape[1]):
            vals.append(structural_similarity(
                gt[i, j, ...], out[i, j, ...], data_range=data_range))
    return median(vals)


def calc_metircs(gt, out, tag):
    ssim, ssimMAP = structural_similarity(gt, out, data_range=1, full=True)
    nrmse = normalized_root_mse(gt, out)
    psnr = peak_signal_noise_ratio(gt, out, data_range=1)
    uqi = UQICalc(gt, out)
    diff = gt - out
    dif_std = np.std(diff)
    metrics = {
        "SSIM"+tag: ssim,
        "NRMSE"+tag: nrmse,
        "PSNR"+tag: psnr,
        "UQI"+tag: uqi,
        "SDofDiff"+tag: dif_std
    }
    return metrics, ssimMAP, diff


def log_images(writer, inputs, outputs, targets, step, section='', imID=0, chID=0):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[imID, chID, ...],
                                      normalize=True,
                                      scale_each=True),
                     step)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                         vutils.make_grid(inputs[imID, chID, ...],
                                          normalize=True,
                                          scale_each=True),
                         step)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                         vutils.make_grid(targets[imID, chID, ...],
                                          normalize=True,
                                          scale_each=True),
                         step)


def SaveNIFTI(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)


class DataSpaceHandler:
    def __init__(self, **kwargs) -> None:
        self.dataspace_inp = kwargs['dataspace_inp']
        self.model_dataspace_inp = kwargs['model_dataspace_inp']
        self.dataspace_gt = kwargs['dataspace_gt']
        self.model_dataspace_gt = kwargs['model_dataspace_gt']
        self.dataspace_out = kwargs['dataspace_out']
        self.model_dataspace_out = kwargs['model_dataspace_out']
        self.data_dim = kwargs['inplane_dims']
        self.fftnorm = kwargs['fftnorm']

    def getTransforms(self):
        return getDataSpaceTransforms(self.dataspace_inp, self.model_dataspace_inp, self.dataspace_gt, self.model_dataspace_gt)


class DataHandler:
    def __init__(self, dataspace_op: DataSpaceHandler, inp=None, gt=None, out=None):
        self.dataspace_op = dataspace_op
        self.inp = inp
        self.gt = gt
        self.out = out
        self.inpK = None
        self.gtK = None
        self.outK = None
        self.outCorrectedK = None

    def setInpK(self, x):
        self.inpK = x

    def setGTK(self, x):
        self.gtK = x

    def setOutK(self, x):
        self.outK = x

    def setOutCorrectedK(self, x):
        self.outCorrectedK = x

    # Get kspace

    def __getK(self, x, dataspace, k, imnorm=False):
        if k is not None:
            return k
        elif dataspace == 1 or x is None:
            return x
        else:
            return fftNc(data=x if not imnorm else x/x.max(), dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    def getKInp(self, imnorm=False):
        return self.__getK(self.inp, self.dataspace_op.model_dataspace_inp, self.inpK, imnorm)

    def getKGT(self, imnorm=False):
        return self.__getK(self.gt, self.dataspace_op.model_dataspace_gt, self.gtK, imnorm)

    def getKOut(self, imnorm=False):
        return self.__getK(self.out, self.dataspace_op.dataspace_out, self.outK, imnorm)

    def getKOutCorrected(self):
        return self.outCorrectedK

    # Get Image space

    def __getIm(self, x, dataspace):
        if dataspace == 0 or x is None:
            return x
        else:
            return ifftNc(data=x, dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    def getImInp(self):
        return self.__getIm(self.inp, self.dataspace_op.model_dataspace_inp)

    def getImGT(self):
        return self.__getIm(self.gt, self.dataspace_op.model_dataspace_gt)

    def getImOut(self):
        return self.__getIm(self.out, self.dataspace_op.dataspace_out)

    def getImOutCorrected(self):
        if self.outCorrectedK is None:
            return None
        else:
            return ifftNc(data=self.outCorrectedK, dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    # all combo

    def getKAll(self):
        return (self.getKInp(), self.getKGT(), self.getKOut())

    def getImgAll(self):
        return (self.getImInp(), self.getImGT(), self.getImOut())


class ResSaver():
    def __init__(self, out_path, save_inp=False, do_norm=False):
        self.out_path = out_path
        self.save_inp = save_inp
        self.do_norm = do_norm

    def CalcNSave(self, datumHandler: DataHandler, outfolder, datacon_operator: DataConsistency = None):
        outpath = os.path.join(self.out_path, outfolder)
        os.makedirs(outpath, exist_ok=True)

        inp = datumHandler.getImInp().float().numpy()
        out = datumHandler.getImOut().float().numpy()
        SaveNIFTI(out, os.path.join(outpath, "out.nii.gz"))

        if self.save_inp:
            SaveNIFTI(inp, os.path.join(outpath, "inp.nii.gz"))

        gt = datumHandler.getImGT()
        if gt is not None:
            gt = gt.float().numpy()

            if self.do_norm:
                out = out/out.max()
                inp = inp/inp.max()
                gt = gt/gt.max()

            out_metrics, out_ssimMAP, out_diff = calc_metircs(
                gt, out, tag="Out")
            SaveNIFTI(out_ssimMAP, os.path.join(outpath, "ssimMAPOut.nii.gz"))
            SaveNIFTI(out_diff, os.path.join(outpath, "diffOut.nii.gz"))

            inp_metrics, inp_ssimMAP, inp_diff = calc_metircs(
                gt, inp, tag="Inp")
            SaveNIFTI(inp_ssimMAP, os.path.join(outpath, "ssimMAPInp.nii.gz"))
            SaveNIFTI(inp_diff, os.path.join(outpath, "diffInp.nii.gz"))

            metrics = {**out_metrics, **inp_metrics}
        else:
            metrics = None

        if datacon_operator is not None:
            datumHandler.setOutCorrectedK(datacon_operator.apply(out_ksp=datumHandler.getKOut(
                imnorm=True), full_ksp=datumHandler.getKGT(imnorm=True), under_ksp=datumHandler.inpK))  # TODO: param for imnorm
            # outCorrected = abs(datumHandler.getImOutCorrected()).float().numpy()
            outCorrected = datumHandler.getImOutCorrected(
            ).real.float().numpy()  # TODO: param real v abs
            SaveNIFTI(out, os.path.join(outpath, "outCorrected.nii.gz"))
            if gt is not None:
                # if self.do_norm:
                #     outCorrected = outCorrected/outCorrected.max()
                outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(
                    gt, outCorrected, tag="OutCorrected")
                SaveNIFTI(outCorrected_ssimMAP, os.path.join(
                    outpath, "ssimMAPOutCorrected.nii.gz"))
                SaveNIFTI(outCorrected_diff, os.path.join(
                    outpath, "diffOutCorrected.nii.gz"))
                metrics = {**metrics, **outCorrected_metrics}

        return metrics


def CustomInitialiseWeights(m):
    """Initialises Weights for our networks.
    Currently it's only for Convolution and Batch Normalisation"""

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if type(m.weight) is torch.nn.ParameterList or m.weight.dtype is torch.cfloat:
            torchcomplex.nn.init.trabelsi_standard_(m.weight, kind="glorot")
        else:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# meant for converting the old resnet model Resnet2Dv2b14 (used in v0 pipeline) to the new ReconResNet model


def ConvertCheckpoint(checkpoint_path, new_checkpoint_path, newModel):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    del checkpoint['model']
    old_state_dict = checkpoint['state_dict']
    new_keys = list(newModel.state_dict().keys())
    new_state_dict = OrderedDict(
        [(new_keys[i], v) for i, (k, v) in enumerate(old_state_dict.items())])
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, new_checkpoint_path)
