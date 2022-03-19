import os
import sys
from argparse import ArgumentParser
from os.path import join as pjoin
from statistics import median
from typing import Any, List
import collections

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torchio as tio
from Bridge.WarpDrives.PDUNet.pd import PrimalDualNetwork
from Bridge.WarpDrives.PDUNet.pd2 import PrimalDualNetwork as PrimalDualNetworkNoResidue
from Bridge.WarpDrives.PDUNet.pd_crelu import PrimalDualNetwork as PrimalDualNetworkCReLU
from Bridge.WarpDrives.PDUNet.pd_crelu_largekern import PrimalDualNetwork as PrimalDualNetworkCReLUMegaKern
from Bridge.WarpDrives.ReconResNet.ReconResNet import ResNet
from Bridge.WarpDrives.ReconResNet.DualSpaceReconResNet import DualSpaceResNet
from Bridge.WarpDrives.ReconResNet.CentreKSPMoCoNet import CentreKSPMoCoNet
from Bridge.WarpDrives.ReconResNet.MoCoReCo import MoCoReCoNet
from Engineering.data_consistency import DataConsistency
from Engineering.datasets.fastMRI import createFastMRIDS
from Engineering.datasets.medfile import createFileDS
from Engineering.datasets.tio import create_patchQs, createTIOSubDS
from Engineering.pLoss.perceptual_loss import PerceptualLoss
from Engineering.transforms import augmentations as pytAugmentations
from Engineering.transforms import motion as pytMotion
from Engineering.transforms import transforms as pytTransforms
from Engineering.transforms.tio import augmentations as tioAugmentations
from Engineering.transforms.tio import motion as tioMotion
from Engineering.transforms.tio import transforms as tioTransforms
from Engineering.utilities import (CustomInitialiseWeights, DataHandler,
                                   DataSpaceHandler, ResSaver, fetch_vol_subds, fetch_vol_subds_fastMRI, getSSIM,
                                   log_images, process_slicedict, process_testbatch)
from pytorch_lightning.core.lightning import LightningModule
from pytorch_msssim import MS_SSIM, SSIM
from torch import nn
from torch.utils.data.dataloader import DataLoader


class ReconEngine(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.hparams.cuda else "cpu")

        if self.hparams.modelID == 0:
            self.net = ResNet(in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels, res_blocks=self.hparams.model_res_blocks,
                              starting_nfeatures=self.hparams.model_starting_nfeatures, updown_blocks=self.hparams.model_updown_blocks,
                              is_relu_leaky=self.hparams.model_relu_leaky, do_batchnorm=self.hparams.model_do_batchnorm,
                              res_drop_prob=self.hparams.model_drop_prob, is_replicatepad=self.hparams.model_is_replicatepad, out_act=self.hparams.model_out_act, forwardV=self.hparams.model_forwardV,
                              upinterp_algo=self.hparams.model_upinterp_algo, post_interp_convtrans=self.hparams.model_post_interp_convtrans, is3D=self.hparams.is3D)  # TODO think of 2D
            # self.net = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        elif self.hparams.modelID == 2:
            self.net = DualSpaceResNet(in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels, res_blocks=self.hparams.model_res_blocks,
                                       starting_nfeatures=self.hparams.model_starting_nfeatures, updown_blocks=self.hparams.model_updown_blocks,
                                       is_relu_leaky=self.hparams.model_relu_leaky, do_batchnorm=self.hparams.model_do_batchnorm,
                                       res_drop_prob=self.hparams.model_drop_prob, is_replicatepad=self.hparams.model_is_replicatepad, out_act=self.hparams.model_out_act, forwardV=self.hparams.model_forwardV,
                                       upinterp_algo=self.hparams.model_upinterp_algo, post_interp_convtrans=self.hparams.model_post_interp_convtrans, is3D=self.hparams.is3D,
                                       connect_mode=self.hparams.model_dspace_connect_mode, inner_norm_ksp=self.hparams.model_inner_norm_ksp)
        elif self.hparams.modelID == 3:  # Primal-Dual Network, complex Primal
            self.net = PrimalDualNetwork(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=True,
                                         use_original_init=True,
                                         use_complex_primal=True,
                                         g_normtype="magmax",
                                         transform="Fourier",
                                         return_abs=True)
        elif self.hparams.modelID == 4:  # Primal-Dual Network, absolute Primal
            self.net = PrimalDualNetwork(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=True,
                                         use_original_init=True,
                                         use_complex_primal=False,
                                         g_normtype="magmax",
                                         transform="Fourier")
        elif self.hparams.modelID == 5:  # Primal-Dual UNet Network, absolute Primal
            self.net = PrimalDualNetwork(n_primary=4, n_dual=5, n_iterations=2,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=False,
                                         g_normtype="magmax",
                                         transform="Fourier")
        # Primal-Dual Network v2 (no residual), complex Primal - just there for backward compatibility, will be removed.
        elif self.hparams.modelID == 6:
            self.net = PrimalDualNetworkNoResidue(n_primary=5, n_dual=5, n_iterations=10,
                                                  use_original_block=True,
                                                  use_original_init=True,
                                                  use_complex_primal=True,
                                                  residuals=False,
                                                  g_normtype="magmax",
                                                  transform="Fourier",
                                                  return_abs=True)
        # Primal-Dual UNet Network v2 (no residual), absolute Primal - just there for backward compatibility, will be removed.
        elif self.hparams.modelID == 7:
            self.net = PrimalDualNetworkNoResidue(n_primary=4, n_dual=5, n_iterations=2,
                                                  use_original_block=False,
                                                  use_original_init=False,
                                                  use_complex_primal=False,
                                                  residuals=False,
                                                  g_normtype="magmax",
                                                  transform="Fourier",
                                                  return_abs=True)
        elif self.hparams.modelID == 8:  # CentreKSPMoCoNet - Default param as Ale used. Parametereise It!! TODO
            self.net = CentreKSPMoCoNet(updown_blocks=self.hparams.model_updown_blocks,
                                        upinterp_algo=self.hparams.model_upinterp_algo, complex_moconet=False)
        elif self.hparams.modelID == 9:  # CentreKSPMoCoNet - Default param as Ale used. Parametereise It!! TODO
            self.net = CentreKSPMoCoNet(updown_blocks=self.hparams.model_updown_blocks,
                                        upinterp_algo=self.hparams.model_upinterp_algo, complex_moconet=True)
        elif self.hparams.modelID == 10:  # CentreKSPMoCoNet - Default param as Ale used. Parametereise It!! TODO
            self.net = MoCoReCoNet(
                upinterp_algo=self.hparams.model_upinterp_algo, complex_moconet=False)
        elif self.hparams.modelID == 11:  # Primal-Dual UNet Network, complex Primal
            self.net = PrimalDualNetwork(n_primary=4, n_dual=5, n_iterations=2,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=True,
                                         g_normtype="magmax",
                                         transform="Fourier",
                                         return_abs=True)

        ###new CReLU complex models
        elif self.hparams.modelID == 33:  # Primal-Dual Network, complex Primal
            self.net = PrimalDualNetworkCReLU(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=True,
                                         use_original_init=True,
                                         use_complex_primal=True,
                                         g_normtype="magmax",
                                         transform="Fourier",
                                         return_abs=True)
        elif self.hparams.modelID == 44:  # Primal-Dual Network, absolute Primal
            self.net = PrimalDualNetworkCReLU(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=True,
                                         use_original_init=True,
                                         use_complex_primal=False,
                                         g_normtype="magmax",
                                         transform="Fourier")
        elif self.hparams.modelID == 55:  # Primal-Dual UNet Network, absolute Primal
            self.net = PrimalDualNetworkCReLU(n_primary=4, n_dual=5, n_iterations=2,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=False,
                                         g_normtype="magmax",
                                         transform="Fourier")
        elif self.hparams.modelID == 111:  # Primal-Dual UNet Network, complex Primal
            self.net = PrimalDualNetworkCReLU(n_primary=4, n_dual=5, n_iterations=2,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=True,
                                         g_normtype="magmax",
                                         transform="Fourier",
                                         return_abs=True)

        ###new CReLU complex models + 10 Iterations and 5 primaries
        elif self.hparams.modelID == 555:  # Primal-Dual UNet Network, absolute Primal
            self.net = PrimalDualNetworkCReLU(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=False,
                                         g_normtype="magmax",
                                         transform="Fourier")
        elif self.hparams.modelID == 1111:  # Primal-Dual UNet Network, complex Primal
            self.net = PrimalDualNetworkCReLU(n_primary=5, n_dual=5, n_iterations=10,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=True,
                                         g_normtype="magmax",
                                         transform="Fourier",
                                         return_abs=True)

        elif self.hparams.modelID == 556:  # Primal-Dual UNet Network, absolute Primal
            self.net = PrimalDualNetworkCReLUMegaKern(n_primary=5, n_dual=5, n_iterations=5,
                                         use_original_block=False,
                                         use_original_init=False,
                                         use_complex_primal=False,
                                         kernel_size=21,
                                         g_normtype="magmax",
                                         transform="Fourier")

        else:
            # TODO: other models
            sys.exit(
                "Only ReconResNet and DualSpaceResNet have been implemented so far in ReconEngine")

        if "custom_step" in dir(self.net): #only this if is to stay, rest are just temporary
            if self.hparams.ds_mode == 2: #fastMRI DS
                if (self.hparams.modelID >= 3 and self.hparams.modelID <= 7) or self.hparams.modelID == 11:
                    self.shared_step = self.custom_shared_step
            elif self.hparams.taskID == 1: #MoCo task
                if self.hparams.modelID >= 8 and self.hparams.modelID <= 10:
                    self.shared_step = self.custom_shared_step
            # self.shared_step = self.custom_shared_step

        if bool(self.hparams.preweights_path):
            print("Pre-weights found, loding...")
            chk = torch.load(self.hparams.preweights_path, map_location='cpu')
            self.net.load_state_dict(chk['state_dict'])

        if self.hparams.lossID == 0:
            if self.hparams.in_channels != 1 or self.hparams.out_channels != 1:
                sys.exit(
                    "Perceptual Loss used here only works for 1 channel input and output")
            self.loss = PerceptualLoss(device=device, loss_model=self.hparams.ploss_model, resize=None,
                                       loss_type=self.hparams.ploss_type, n_level=self.hparams.ploss_level)
        elif self.hparams.lossID == 1:
            self.loss = nn.L1Loss(reduction='mean')
        elif self.hparams.lossID == 2:
            self.loss = MS_SSIM(channel=self.hparams.out_channels, data_range=1,
                                spatial_dims=3 if self.hparams.is3D else 2, nonnegative_ssim=False).to(device)
        elif self.hparams.lossID == 3:
            self.loss = SSIM(channel=self.hparams.out_channels, data_range=1,
                             spatial_dims=3 if self.hparams.is3D else 2, nonnegative_ssim=False).to(device)
        else:
            sys.exit("Invalid Loss ID")

        self.dataspace = DataSpaceHandler(**self.hparams)

        if self.hparams.ds_mode == 0:
            trans = tioTransforms
            augs = tioAugmentations
        elif self.hparams.ds_mode == 1:
            trans = pytTransforms
            augs = pytAugmentations

        # TODO parameterised everything
        self.init_transforms = []
        self.aug_transforms = []
        self.transforms = []
        if self.hparams.ds_mode == 0 and self.hparams.cannonicalResample:  # Only applicable for TorchIO
            self.init_transforms += [tio.ToCanonical(), tio.Resample('gt')]
        if self.hparams.ds_mode == 0 and self.hparams.forceNormAffine:  # Only applicable for TorchIO
            self.init_transforms += [trans.ForceAffine()]
        if self.hparams.croppad and self.hparams.ds_mode == 1:
            self.init_transforms += [
                trans.CropOrPad(size=self.hparams.input_shape)]
        if self.hparams.ds_mode != 2:  # IntensityNorm is not applied for fastMRI DS
            self.init_transforms += [trans.IntensityNorm(
                type=self.hparams.norm_type, return_meta=self.hparams.motion_return_meta)]
        # dataspace_transforms = self.dataspace.getTransforms() #TODO: dataspace transforms are not in use
        # self.init_transforms += dataspace_transforms
        if bool(self.hparams.random_crop) and self.hparams.ds_mode == 1:
            self.aug_transforms += [augs.RandomCrop(
                size=self.hparams.random_crop, p=self.hparams.p_random_crop)]
        if self.hparams.p_contrast_augment > 0:
            self.aug_transforms += [augs.getContrastAugs(
                p=self.hparams.p_contrast_augment)]
        # if the task if MoCo and pre-corrupted vols are not supplied
        if self.hparams.taskID == 1 and not bool(self.hparams.train_path_inp):
            if self.hparams.motion_mode == 0 and self.hparams.ds_mode == 0:
                motion_params = {k.split('motionmg_')[
                    1]: v for k, v in self.hparams.items() if k.startswith('motionmg')}
                self.transforms += [tioMotion.RandomMotionGhostingFast(
                    **motion_params), trans.IntensityNorm()]
            elif self.hparams.motion_mode == 1 and self.hparams.ds_mode == 1 and not self.hparams.is3D:
                self.transforms += [pytMotion.Motion2Dv0(
                    sigma_range=self.hparams.motion_sigma_range, n_threads=self.hparams.motion_n_threads, p=self.hparams.motion_p, return_meta=self.hparams.motion_return_meta)]
            elif self.hparams.motion_mode == 2 and self.hparams.ds_mode == 1 and not self.hparams.is3D:
                self.transforms += [pytMotion.Motion2Dv1(sigma_range=self.hparams.motion_sigma_range, n_threads=self.hparams.motion_n_threads,
                                                         restore_original=self.hparams.motion_restore_original, p=self.hparams.motion_p, return_meta=self.hparams.motion_return_meta)]
            else:
                sys.exit(
                    "Error: invalid motion_mode, ds_mode, is3D combo. Please double check!")

        self.static_metamat = sio.loadmat(self.hparams.static_metamat_file) if bool(
            self.hparams.static_metamat_file) else None
        if self.hparams.taskID == 0 and self.hparams.use_datacon:
            self.datacon = DataConsistency(
                isRadial=self.hparams.is_radial, metadict=self.static_metamat)
        else:
            self.datacon = None

        input_shape = self.hparams.input_shape if self.hparams.is3D else self.hparams.input_shape[
            :-1]
        self.example_input_array = torch.empty(
            self.hparams.batch_size, self.hparams.in_channels, *input_shape).float()
        self.saver = ResSaver(
            self.hparams.res_path, save_inp=self.hparams.save_inp, save_gt=self.hparams.save_gt if "save_gt" in self.hparams else False, do_norm=self.hparams.do_savenorm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }
        if self.hparams.lr_decay_type:  # If this is not zero
            optim_dict["lr_scheduler"] = {
                "scheduler": self.hparams.lrScheduler_func(optimizer, **self.hparams.lrScheduler_param_dict),
                'monitor': 'val_loss',
            }
        return optim_dict

    def forward(self, x):
        return self.net(x)

    def slice_squeeze(self, x):
        return x.squeeze(-1) if self.hparams.ds_mode == 0 and not self.hparams.is3D else x

    def shared_step(self, batch):
        prediction = self(self.slice_squeeze(batch['inp']['data']))
        loss = self.loss(prediction, self.slice_squeeze(
            batch['gt']['data']).to(prediction.dtype))
        if self.hparams.IsNegLoss:
            loss = -loss
        return prediction, loss

    def custom_shared_step(self, batch):
        prediction, loss = self.net.custom_step(batch, self.slice_squeeze, self.loss)
        if self.hparams.IsNegLoss:
            loss = -loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch)
        self.log("running_loss", loss)
        # self.meta_logger("train", batch_idx, {key:val for (key,val) in batch['inp'].items() if "Meta" in key})
        self.img_logger("train", batch_idx, self.slice_squeeze(
            batch['inp']['data']).cpu(), prediction.detach().cpu(), self.slice_squeeze(batch['gt']['data']).cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch)
        gt = self.slice_squeeze(batch['gt']['data']).cpu()
        inp = self.slice_squeeze(batch['inp']['data']).cpu()
        prediction = prediction.detach().cpu()
        ssim = getSSIM(gt.numpy(), prediction.numpy(), data_range=1)
        self.img_logger("val", batch_idx, inp, prediction, gt)
        return {'val_loss': loss, 'val_ssim': ssim}

    def test_step(self, *args):
        prediction, loss = self.shared_step(args[0])
        if not self.hparams.is3D:
            prediction = prediction.unsqueeze(-1)
        if bool(self.hparams.patch_size):
            self.out_aggregators[args[0]['filename'][0]].add_batch(
                prediction.detach().cpu(), args[0][tio.LOCATION])
        else:
            if not self.hparams.is3D and 'sliceID' in args[0]:
                # self.out_aggregators[args[0]['filename'][0]][args[0]['sliceID'][0].item()] = prediction.detach().cpu()
                process_testbatch(self.out_aggregators, args[0], prediction)
            else:
                self.out_aggregators[args[0]['filename']
                                     [0]] = prediction.detach().cpu()
        return {'test_loss': loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).median()
        self.log('training_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).median()
        avg_ssim = np.median(np.stack([x['val_ssim'] for x in outputs]))
        self.log('val_loss', avg_loss)
        self.log('val_ssim', avg_ssim)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # If multiple vols supplied during patch-based testing, outputs will be list of list, otherwise a list of dicts
        if isinstance(outputs, list) and isinstance(outputs[0], list):
            outputs = sum(outputs, [])
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).median()
        self.log('test_loss', avg_loss)
        filenames = self.out_aggregators.keys()
        test_metrics = []
        test_ssim = []
        test_ssim_corrected = []
        for filename in tqdm(filenames):
            try:
                if bool(self.hparams.patch_size):
                    out = self.out_aggregators[filename].get_output_tensor(
                    ).squeeze()
                    sub = self.grid_samplers[filename].subject
                else:
                    if not self.hparams.is3D and type(self.out_aggregators[filename]) is dict:
                        out = process_slicedict(self.out_aggregators[filename]) #TODO: If the model is predicting anything other than Images, this step has to be modified.
                        sub = fetch_vol_subds_fastMRI(self.test_subjectds, filename) if self.hparams.ds_mode==2 else fetch_vol_subds(self.test_subjectds, filename)
                    else:
                        out = self.out_aggregators[filename].squeeze()
                        sub = self.test_subjectds[self.test_filenames.index(
                            filename)]
                    assert sub[
                        'filename'] == filename, "The filename of the test subject doesn't match with the fetched test subject (Index issue!)"
                inp = sub['inp']['data'].squeeze()
                gt = sub['gt']['data'].squeeze()
                if isinstance(out, torch.HalfTensor):
                    out = out.float()  # TODO: find a better way to do this. This might not be a good way
                dHandler = DataHandler(dataspace_op=self.dataspace, inp=inp, gt=gt,
                                    #    out=out, metadict=sub['metadict'] if 'metadict' in sub else None) #Should be actually this. But as metadict is not handled for non-fastMRI DSs for now, the following line is used:
                                    out=out, metadict=sub['metadict'] if ('metadict' in sub) and (self.hparams.ds_mode==2) else None) #TODO: handle meta dict for non-fastMRI DSs
                dHandler.setInpK(sub['inp']['ksp'].squeeze()
                                if "ksp" in sub['inp'] else None)
                dHandler.setGTK(sub['gt']['ksp'].squeeze()
                                if "ksp" in sub['gt'] else None)
                metrics = self.saver.CalcNSave(dHandler, filename.split(".")[
                                            0], datacon_operator=self.datacon)
                if metrics is not None:
                    metrics['file'] = filename
                    test_metrics.append(metrics)
                    test_ssim.append(round(metrics['SSIMOut'], 4))
                    self.log("running_test_ssim", test_ssim[-1])
                    if "SSIMOutCorrected" in metrics:
                        test_ssim_corrected.append(round(metrics['SSIMOutCorrected'], 4))
                        self.log("running_test_ssim_corrected", test_ssim_corrected[-1])
            except Exception as ex:
                print(f"For filename: {filename}, encountered error: {str(ex)}")
        if len(test_metrics) > 0:
            self.log("test_ssim", median(test_ssim))
            if len(test_ssim_corrected) > 0:
                self.log("test_ssim_corrected", median(test_ssim_corrected))
            df = pd.DataFrame.from_dict(test_metrics)
            df.to_csv(pjoin(self.hparams.save_path, self.hparams.run_name,
                      'Results.csv'), index=False)

    def create_dataset(self, split: str) -> tio.SubjectsDataset:
        if split == "train":
            path_gt = self.hparams.train_path_gt
            path_inp = self.hparams.train_path_inp
            filename_filter = self.hparams.train_filename_filter
        elif split == "val":
            path_gt = self.hparams.val_path_gt
            path_inp = self.hparams.val_path_inp
            filename_filter = self.hparams.val_filename_filter
        elif split == "test":
            path_gt = self.hparams.test_path_gt
            path_inp = self.hparams.test_path_inp
            filename_filter = self.hparams.test_filename_filter
        params = {"root_gt": path_gt, "root_input": path_inp, "filename_filter": filename_filter,
                  "split_csv": self.hparams.split_csv, "split": split, "data_mode": self.hparams.file_type,
                  "isKSpace": False, "isGTNonImg": self.hparams.isGTNonImg, "init_transforms": self.init_transforms,
                  "aug_transforms": self.aug_transforms if split != "test" else None, "transforms": self.transforms}  # TODO: need to implement for kSpace
        if self.hparams.ds_mode == 0:
            dataset, filenames = createTIOSubDS(**params)
        elif self.hparams.ds_mode == 1:
            params["is3D"] = self.hparams.is3D
            params["mid_n"] = self.hparams.ds2D_mid_n
            params["mid_per"] = self.hparams.ds2D_mid_per
            params["random_n"] = self.hparams.ds2D_random_n
            if "processed_csv" in self.hparams and bool(self.hparams.processed_csv):
                params["processed_csv"] = self.hparams.processed_csv
            dataset, filenames = createFileDS(**params)
        elif self.hparams.ds_mode == 2:
            del params["data_mode"], params["isKSpace"], params["isGTNonImg"]
            params["combine_coils"] = self.hparams.combine_coils
            params["complex_image_modes"] = self.hparams.complex_image_modes
            params["fastMRI_challenge"] = self.hparams.fastMRI_challenge
            params["use_dataset_cache"] = self.hparams.use_dataset_cache
            params["sample_rate"] = self.hparams.sample_rate
            params["volume_sample_rate"] = self.hparams.volume_sample_rate
            params["num_cols"] = self.hparams.num_cols
            params["mask_type"] = self.hparams.mask_type
            params["center_fractions"] = self.hparams.center_fractions
            params["accelerations"] = self.hparams.accelerations
            if "processed_csv" in self.hparams and bool(self.hparams.processed_csv):
                params["processed_csv"] = self.hparams.processed_csv
            dataset, filenames = createFastMRIDS(**params)

        if bool(self.hparams.patch_size):
            if self.hparams.ds_mode == 0:
                if split == "train":
                    dataset, _, _ = create_patchQs(
                        train_subs=dataset, val_subs=None, patch_size=self.hparams.patch_size,
                        patch_qlen=self.hparams.patch_qlen, patch_per_vol=self.hparams.patch_per_vol,
                        inference_strides=self.hparams.patch_inference_strides)
                elif split == "val":
                    _, dataset, _ = create_patchQs(
                        train_subs=None, val_subs=dataset, patch_size=self.hparams.patch_size,
                        patch_qlen=self.hparams.patch_qlen, patch_per_vol=self.hparams.patch_per_vol,
                        inference_strides=self.hparams.patch_inference_strides)
                elif split == "test":
                    _, _, dataset = create_patchQs(
                        train_subs=None, val_subs=dataset, patch_size=self.hparams.patch_size,
                        patch_qlen=self.hparams.patch_qlen, patch_per_vol=self.hparams.patch_per_vol,
                        inference_strides=self.hparams.patch_inference_strides)
            else:
                sys.exit(
                    "Error: patch_size can only be used with ds_mode 0 (TorchIO)")
        if split == "test":
            return dataset, filenames
        else:
            return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(split="train"),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.create_dataset(split="val"),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        ds, filenames = self.create_dataset(split="test")
        if bool(self.hparams.patch_size):
            self.grid_samplers = {}
        else:  # If its not patch-based, then it will only return one subject dataset instead of a list of grid_samplers
            self.test_subjectds = ds
            ds = [ds]
        test_loaders = []
        self.out_aggregators = collections.defaultdict(dict)
        self.test_filenames = filenames
        for grid_sampler in ds:
            test_loaders.append(DataLoader(grid_sampler,
                                shuffle=False,
                                batch_size=self.hparams.batch_size,
                                pin_memory=True, num_workers=self.hparams.num_workers))
            if bool(self.hparams.patch_size):
                self.out_aggregators[grid_sampler.subject.filename] = tio.inference.GridAggregator(
                    grid_sampler, overlap_mode="average")
                self.grid_samplers[grid_sampler.subject.filename] = grid_sampler
        return test_loaders

    def img_logger(self, tag, batch_idx, inp=None, pred=None, gt=None) -> None:
        if self.hparams.tbactive and self.hparams.im_log_freq > -1 and batch_idx % self.hparams.im_log_freq == 0:
            if len(inp.shape) == 5:  # 3D
                central_slice = inp.shape[-2] // 2
                inp = inp[:, :, central_slice]
                pred = pred[:, :, central_slice]
                gt = gt[:, :, central_slice]
            log_images(self.logger[-1].experiment, inp if not torch.is_complex(inp) else torch.abs(inp),
                       pred if not torch.is_complex(pred) else torch.abs(pred), gt if not torch.is_complex(gt) else torch.abs(gt), batch_idx, tag)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nothing_yet', type=str)
        return parser
