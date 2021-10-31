from argparse import ArgumentParser
import os
import sys
from os.path import join as pjoin
from statistics import median
from typing import Any, List
import numpy as np

import pandas as pd
import torch
import torchio as tio
from Bridge.WarpDrives.ReconResNet.ReconResNet import ResNet
from Bridge.WarpDrives.ReconResNet.Resnet2Dv2b14 import ResNet as ResNetHHPaper
from Engineering.datasets.tio import create_patchQs, createTIOSubDS
from Engineering.pLoss.perceptual_loss import PerceptualLoss
from Engineering.transforms.tio.augmentations import (
    AdaptiveHistogramEqualization, AdjustGamma, AdjustLog, AdjustSigmoid, getContrastAugs)
from Engineering.transforms.tio.transforms import (ForceAffine, IntensityNorm,
                                                   RandomMotionGhostingV2, getDataSpaceTransforms)
from Engineering.utilities import CustomInitialiseWeights, DataHandler, DataSpaceHandler, ResSaver, getSSIM, log_images
from pytorch_lightning.core.lightning import LightningModule
from pytorch_ssim import SSIM
# from pytorch_msssim import SSIM, MSSSIM
from torch import nn
from torch.utils.data.dataloader import DataLoader
from Engineering.data_consistency import DataConsistency
import scipy.io as sio

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
                              res_drop_prob=self.hparams.model_drop_prob, is_replicatepad=1, out_act="sigmoid", forwardV=self.hparams.model_forwardV,
                              upinterp_algo=self.hparams.model_upinterp_algo, post_interp_convtrans=self.hparams.model_post_interp_convtrans, is3D=self.hparams.is3D)  # TODO think of 2D
            # self.net = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        elif self.hparams.modelID == 1:
            self.net = ResNetHHPaper()
            self.net.apply(CustomInitialiseWeights)
        else:
            # TODO: other models
            sys.exit("Only ResNet has been implemented so far in ReconEngine")

        if self.hparams.lossID == 0:
            if self.hparams.in_channels != 1 or self.hparams.out_channels != 1:
                sys.exit(
                    "Perceptual Loss used here only works for 1 channel input and output")
            self.loss = PerceptualLoss(device=device, loss_model="unet3Dds", resize=None,
                                       loss_type=self.hparams.ploss_type, n_level=self.hparams.ploss_level)  # TODO thinkof 2D
        elif self.hparams.lossID == 1:
            self.loss = nn.L1Loss(reduction='mean')
        elif self.hparams.lossID == 2:
            self.loss = MSSSIM(channel=self.hparams.out_channels).to(device)
        elif self.hparams.lossID == 3:
            self.loss = SSIM(channel=self.hparams.out_channels).to(device)
        else:
            sys.exit("Invalid Loss ID")

        self.dataspace = DataSpaceHandler(**self.hparams)

        # TODO parameterised everything
        self.init_transforms = []
        self.aug_transforms = []
        self.transforms = []
        if self.hparams.cannonicalResample:
            self.init_transforms += [tio.ToCanonical(), tio.Resample('gt')]
        if self.hparams.forceNormAffine:
            self.init_transforms += [ForceAffine()]
        self.init_transforms += [IntensityNorm()]
        dataspace_transforms = self.dataspace.getTransforms()
        if self.hparams.contrast_augment:
            self.aug_transforms += [getContrastAugs()]
        if self.hparams.taskID == 1 and not bool(self.hparams.train_path_inp): #if the task if MoCo and pre-corrupted vols are not supplied
            self.transforms += [RandomMotionGhostingV2(), IntensityNorm()] 

        self.static_metamat = sio.loadmat(self.hparams.static_metamat_file) if bool(self.hparams.static_metamat_file) else None
        if self.hparams.taskID == 0 and self.hparams.use_datacon:
            self.datacon = DataConsistency(isRadial=self.hparams.is_radial, metadict=self.static_metamat)
        else:
            self.datacon = None

        input_shape = self.hparams.input_shape if self.hparams.is3D else self.hparams.input_shape[:-1]
        self.example_input_array = torch.empty(
            self.hparams.batch_size, self.hparams.in_channels, *input_shape).float()
        self.saver = ResSaver(
            self.hparams.res_path, save_inp=self.hparams.save_inp, do_norm=self.hparams.do_savenorm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }
        if self.hparams.lr_decay_type: #If this is not zero
            optim_dict["lr_scheduler"] = {
                "scheduler": self.hparams.lrScheduler_func(optimizer, **self.hparams.lrScheduler_param_dict),
                'monitor': 'val_loss',
            }
        return optim_dict

    def forward(self, x):
        return self.net(x)

    def slice_squeeze(self, x):
        return x if self.hparams.is3D else x.squeeze(-1)

    def shared_step(self, batch):
        prediction = self(self.slice_squeeze(batch['inp']['data']))
        loss = self.loss(prediction, self.slice_squeeze(batch['gt']['data']))
        if self.hparams.IsNegLoss:
            loss = -loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch)
        self.log("running_loss", loss)
        self.img_logger("train", batch_idx, self.slice_squeeze(batch['inp']['data']), prediction, self.slice_squeeze(batch['gt']['data']))
        return loss
    def validation_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch)
        ssim = getSSIM(self.slice_squeeze(batch['gt']['data']).cpu().numpy(),
                       prediction.detach().cpu().numpy(), data_range=1)
        self.img_logger("val", batch_idx, self.slice_squeeze(batch['inp']['data']), prediction, self.slice_squeeze(batch['gt']['data']))
        return {'val_loss': loss, 'val_ssim': ssim}

    def test_step(self, *args):
        prediction, loss = self.shared_step(args[0])
        if not self.hparams.is3D:
            prediction = prediction.unsqueeze(-1)
        if bool(self.hparams.patch_size):
            self.patch_aggregators[args[0]['filename'][0]].add_batch(
                prediction.detach().cpu(), args[0][tio.LOCATION])
        else:
            sys.exit("Test step only implemented for patching mode")
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
        if isinstance(outputs, list) and isinstance(outputs[0], list): #If multiple vols supplied during patch-based testing, outputs will be list of list, otherwise a list of dicts
            outputs = sum(outputs, [])
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).median()
        self.log('test_loss', avg_loss)
        filenames = self.patch_aggregators.keys()
        test_metrics = []
        test_ssim = []
        for filename in filenames:
            out = self.patch_aggregators[filename].get_output_tensor().squeeze()
            if isinstance(out, torch.HalfTensor):
                out = out.float() #TODO: find a better way to do this. This might not be a good way
            inp = self.grid_samplers[filename].subject['inp'][tio.DATA].squeeze()
            gt = self.grid_samplers[filename].subject['gt'][tio.DATA].squeeze()
            dHandler = DataHandler(dataspace_op=self.dataspace, inp=inp, gt=gt, out=out)
            dHandler.setInpK(self.grid_samplers[filename].subject['inpK'][tio.DATA].squeeze() if "inpK" in self.grid_samplers[filename].subject else None)
            dHandler.setGTK(self.grid_samplers[filename].subject['gtK'][tio.DATA].squeeze() if "gtK" in self.grid_samplers[filename].subject else None)
            metrics = self.saver.CalcNSave(dHandler, filename.split(".")[0], datacon_operator=self.datacon)
            if metrics is not None:
                metrics['file'] = filename
                test_metrics.append(metrics)
                test_ssim.append(round(metrics['SSIMOut'], 4))
                self.log("running_test_ssim", test_ssim[-1])
        if len(test_metrics) > 0:
            self.log("test_ssim", median(test_ssim))
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
        dataset = createTIOSubDS(root_gt=path_gt, root_input=path_inp, filename_filter=filename_filter, 
                                 split_csv=self.hparams.split_csv, split=split, data_mode=self.hparams.file_type,
                                 isKSpace=False, isGTNonImg=self.hparams.isGTNonImg, init_transforms=self.init_transforms,
                                 aug_transforms=self.aug_transforms if split != "test" else None, transforms=self.transforms)  # TODO: need to implement for kSpace
        if bool(self.hparams.patch_size):
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
        ds = self.create_dataset(split="test")
        if bool(self.hparams.patch_size):
            self.patch_aggregators = {}
            self.grid_samplers = {}
        else:  # If its not patch-based, then it will only return one subject dataset instead of a list of grid_samplers
            ds = [ds]
        test_loaders = []
        for grid_sampler in ds:
            test_loaders.append(DataLoader(grid_sampler,
                                shuffle=False,
                                batch_size=self.hparams.batch_size,
                                pin_memory=True, num_workers=self.hparams.num_workers))
            if bool(self.hparams.patch_size):
                self.patch_aggregators[grid_sampler.subject.filename] = tio.inference.GridAggregator(
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
            log_images(self.logger[-1].experiment, inp,
                       pred.detach(), gt, batch_idx, tag)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nothing_yet', type=str)
        return parser
