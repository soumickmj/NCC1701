import argparse
import os
from collections import OrderedDict
from statistics import median
import pandas as pd

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
from Engineering.math.misc import minmax
from Engineering.transforms.tio.transforms import getDataSpaceTransforms


def BoolArgs(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Can be supplied as: ['yes', 'true', 't', 'y', '1'] or ['no', 'false', 'f', 'n', '0']")


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


def calc_metircs(gt, out, tag, norm4diff=False):
    ssim, ssimMAP = structural_similarity(gt, out, data_range=1, full=True)
    nrmse = normalized_root_mse(gt, out)
    psnr = peak_signal_noise_ratio(gt, out, data_range=1)
    uqi = UQICalc(gt, out)
    if norm4diff:
        gt = minmax(gt)
        out = minmax(out)
    diff = gt - out
    dif_std = np.std(diff)
    metrics = {
        "SSIM"+tag: ssim,
        "NRMSE"+tag: nrmse,
        "PSNR"+tag: psnr,
        "UQI"+tag: uqi,
        "SDofDiff"+tag: dif_std
    }
    return metrics, ssimMAP, abs(diff)


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


def ReadNIFTI(file_path):
    """Read a NIFTI file using given file path to an array
    Using: NiBabel"""
    nii = nib.load(file_path)
    return np.array(nii.get_fdata())


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
    def __init__(self, dataspace_op: DataSpaceHandler, inp=None, gt=None, out=None, metadict=None, storeAsTensor=True):
        self.dataspace_op = dataspace_op
        self.storeAsTensor = storeAsTensor
        self.inp = self.__convert_type(inp)
        self.gt = self.__convert_type(gt)
        self.out = self.__convert_type(out)
        self.metadict = metadict
        self.inpK = None
        self.gtK = None
        self.outK = None
        self.outCorrectedK = None

    def __convert_type(self, x):
        if x is None or self.storeAsTensor == torch.is_tensor(x):
            return x
        elif self.storeAsTensor and not torch.is_tensor(x):
            return torch.from_numpy(x)
        else:
            return x.numpy()

    def setInpK(self, x):
        self.inpK = self.__convert_type(x)

    def setGTK(self, x):
        self.gtK = self.__convert_type(x)

    def setOutK(self, x):
        self.outK = self.__convert_type(x)

    def setOutCorrectedK(self, x):
        self.outCorrectedK = self.__convert_type(x)

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
    def __init__(self, out_path, save_inp=False, save_gt=False, do_norm=False):
        self.out_path = out_path
        self.save_inp = save_inp
        self.save_gt = save_gt
        self.do_norm = do_norm

    def CalcNSave(self, datumHandler: DataHandler, outfolder, datacon_operator: DataConsistency = None):
        outpath = os.path.join(self.out_path, outfolder)
        os.makedirs(outpath, exist_ok=True)

        inp = datumHandler.getImInp()
        if torch.is_complex(inp):
            inp = abs(inp)
        inp = inp.float().numpy()

        out = datumHandler.getImOut()
        if torch.is_complex(out):
            out = abs(out)
        out = out.float().numpy()

        SaveNIFTI(out, os.path.join(outpath, "out.nii.gz"))

        if self.save_inp:
            SaveNIFTI(inp, os.path.join(outpath, "inp.nii.gz"))

        gt = datumHandler.getImGT()
        if gt is not None:
            if torch.is_complex(gt):
                gt = abs(gt)
            gt = gt.float().numpy()

            if self.save_gt:
                SaveNIFTI(gt, os.path.join(outpath, "gt.nii.gz"))

            if self.do_norm:
                out = minmax(out)
                inp = minmax(inp)
                gt = minmax(gt)

            out_metrics, out_ssimMAP, out_diff = calc_metircs(
                gt, out, tag="Out", norm4diff=not self.do_norm)
            SaveNIFTI(out_ssimMAP, os.path.join(outpath, "ssimMAPOut.nii.gz"))
            SaveNIFTI(out_diff, os.path.join(outpath, "diffOut.nii.gz"))

            inp_metrics, inp_ssimMAP, inp_diff = calc_metircs(
                gt, inp, tag="Inp", norm4diff=not self.do_norm)
            SaveNIFTI(inp_ssimMAP, os.path.join(outpath, "ssimMAPInp.nii.gz"))
            SaveNIFTI(inp_diff, os.path.join(outpath, "diffInp.nii.gz"))

            metrics = {**out_metrics, **inp_metrics}
        else:
            metrics = None

        if datacon_operator is not None:
            datumHandler.setOutCorrectedK(datacon_operator.apply(out_ksp=datumHandler.getKOut(
                imnorm=True), full_ksp=datumHandler.getKGT(imnorm=True), under_ksp=datumHandler.inpK, metadict=datumHandler.metadict))  # TODO: param for imnorm
            # outCorrected = abs(datumHandler.getImOutCorrected()).float().numpy()
            outCorrected = datumHandler.getImOutCorrected(
            ).real.float().numpy()  # TODO: param real v abs
            SaveNIFTI(outCorrected, os.path.join(
                outpath, "outCorrected.nii.gz"))
            if gt is not None:
                # if self.do_norm:
                #     outCorrected = minmax(outCorrected)
                outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(
                    gt, outCorrected, tag="OutCorrected", norm4diff=not self.do_norm)
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
    assert len(checkpoint['model'].state_dict().keys()) == len(
        newModel.state_dict().keys()), "Number of Params in New and Old models do not match"
    del checkpoint['model']
    old_state_dict = checkpoint['state_dict']
    new_keys = list(newModel.state_dict().keys())
    new_state_dict = OrderedDict(
        [(new_keys[i], v) for i, (k, v) in enumerate(old_state_dict.items())])
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, new_checkpoint_path)


def process_testbatch(out_aggregators, datum, prediction):
    for i in range(len(datum['filename'])):
        out_aggregators[datum['filename'][i]][datum['sliceID']
                                              [i].item()] = prediction[i].detach().cpu()


def process_slicedict(dict_sliceout, axis=-1):
    sliceIDs = sorted(list(dict_sliceout.keys()))
    out = []
    for s in sliceIDs:
        out.append(dict_sliceout[s].squeeze())
    if torch.is_tensor(out[0]):
        return torch.stack(out, axis=axis)
    else:
        return np.stack(out, axis=axis)


def fetch_vol_subds(subjectds, filename, slcaxis=-1):
    df = subjectds.df
    ids = np.array(df.index[df['filename'] == filename].tolist())
    sliceIDs = df[df['filename'] == filename].sliceID.tolist()
    ids = ids[np.argsort(sliceIDs)]
    inp = []
    gt = []
    for i in ids:
        inp.append(subjectds[i]['inp']['data'].squeeze())
        gt.append(subjectds[i]['gt']['data'].squeeze())
    sub = {
        "inp": {
            "data": np.stack(inp, axis=slcaxis)
        },
        "gt": {
            "data": np.stack(gt, axis=slcaxis)
        },
        "filename": filename
    }
    return sub
    # else:
    #     return torch.stack(inp, axis=slcaxis), torch.stack(gt, axis=slcaxis)


def fetch_vol_subds_fastMRI(subjectds, filename, slcaxis=-1):
    df = pd.DataFrame(subjectds.examples, columns=[
                      "fname", "dataslice", "metadata"])
    df["fname"] = df["fname"].apply(lambda x: os.path.basename(x))
    ids = np.array(df.index[df['fname'] == filename].tolist())
    sliceIDs = df[df['fname'] == filename].dataslice.tolist()
    ids = ids[np.argsort(sliceIDs)]
    inp = []
    inpK = []
    gt = []
    gtK = []
    mask = []
    fastMRIAttrs = []
    for i in ids:
        ds = subjectds[i]
        inp.append(ds['inp']['data'].squeeze())
        inpK.append(ds['inp']['ksp'].squeeze())
        gt.append(ds['gt']['data'].squeeze())
        gtK.append(ds['gt']['ksp'].squeeze())
        if "mask" in ds['metadict']:
            mask.append(ds['metadict']['mask'].squeeze(0))
        if "fastMRIAttrs" in ds['metadict']:
            fastMRIAttrs.append(ds['metadict']['fastMRIAttrs'])
    sub = {
        "inp": {
            "data": np.stack(inp, axis=slcaxis),
            "ksp": np.stack(inpK, axis=slcaxis)
        },
        "gt": {
            "data": np.stack(gt, axis=slcaxis),
            "ksp": np.stack(gtK, axis=slcaxis)
        },
        "filename": filename
    }
    if len(mask) > 0 and len(fastMRIAttrs) > 0:
        sub["metadict"] = {
            "mask": np.stack(mask, axis=slcaxis),
            "fastMRIAttrs": fastMRIAttrs
        }
    else:
        if len(mask) > 0:
            sub["metadict"] = {"mask": np.stack(mask, axis=slcaxis)}
        elif len(fastMRIAttrs) > 0:
            sub["metadict"] = {"fastMRIAttrs": fastMRIAttrs}
    return sub
    # else:
    #     return torch.stack(inp, axis=slcaxis), torch.stack(gt, axis=slcaxis)

# class MetaLogger():
#     def __init__(self, active=True) -> None:
#         self.activate = active

#     def __call__(self, tag, batch_idx, metas):
#         if self.active:


#     metas = {key:val for (key,val) in batch['inp'].items() if "Meta" in key}
