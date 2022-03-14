import json
import os
import sys
from os.path import join as pjoin

import tensorboard
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import optim

from .AuxiliaryEngines.ReconEngine import ReconEngine


class Engine(object):

    def __init__(self, parser):

        _temp_args = parser.parse_args()
        if _temp_args.taskID == 0 or _temp_args.taskID == 1:
            parser = ReconEngine.add_model_specific_args(parser)
        else:
            # TODO: implement other engines
            sys.exit(
                "Only ReconEngine has been implemented yet, so only Undersampled Recon or MoCo is possible.")

        parser = Trainer.add_argparse_args(parser)
        hparams = parser.parse_args()

        with open(hparams.datajson_path, 'r') as json_file:
            json_data = json.load(json_file)
        hparams.__dict__.update(json_data)

        hparams.run_name = hparams.run_prefix + "_" + \
            hparams.trainID if bool(hparams.run_prefix) else hparams.trainID
        hparams.res_path = pjoin(
            hparams.save_path, hparams.run_name, "Results")
        os.makedirs(hparams.res_path, exist_ok=True)

        if (hparams.lossID == 0 and hparams.ploss_type == "L1") or (hparams.lossID == 1):
            hparams.IsNegLoss = False
        else:
            hparams.IsNegLoss = True

        if hparams.run_mode == 1 or hparams.run_mode == 4:
            hparams.do_val = True
        else:
            hparams.do_val = False

        if bool(hparams.patch_size):
            l, w, d = hparams.patch_size.split(',')
            hparams.patch_size = (int(l), int(w), int(d))
            hparams.input_shape = hparams.patch_size
        else:
            in_shape = hparams.input_shape.split(',')
            if len(in_shape) == 2:
                in_shape = (int(in_shape[0]), int(in_shape[1]), 1)
            hparams.input_shape = (int(in_shape[0]), int(
                in_shape[1]), int(in_shape[2]))

        if hparams.lr_decay_type == 1:
            hparams.lrScheduler_func = optim.lr_scheduler.StepLR
            hparams.lrScheduler_param_dict = {"step_size": hparams.lr_decay_nepoch,
                                              "gamma": hparams.lr_decay_rate}
            # "last_epoch": last_epoch if args.resume else -1}
        elif hparams.lr_decay_type == 2:
            hparams.lrScheduler_func = optim.lr_scheduler.ReduceLROnPlateau
            hparams.lrScheduler_param_dict = {"factor": hparams.lr_decay_rate,
                                              "mode": "min",  # TODO: args. Min for decreasing loss, max for increasing
                                              # TODO: args. Number of epochs with no improvement after which learning rate will be reduced.
                                              "patience": 10,
                                              # TODO: args. Threshold for measuring the new optimum, to only focus on significant changes.
                                              "threshold": 1e-4,
                                              # TODO: args. Number of epochs to wait before resuming normal operation after lr has been reduced.
                                              "cooldown": 0,
                                              "min_lr": 0}  # TODO: args. A lower bound on the learning rate of all param groups or each group respectively.

        if not hparams.non_deter:
            seed_everything(hparams.seed, workers=True)

        if hparams.resume:
            if hparams.load_test_ckpt:
                print("Loading trained model for testing....")
                path2chk = hparams.test_trainedmodel
            else:
                path2chk = pjoin(hparams.save_path, hparams.run_name)
            if hparams.load_best:
                checkpoint_dir = pjoin(path2chk, "Checkpoints")
                available_checkpoints = {int(c.split("epoch=")[1].split("-")[0]): c for c in [x for x in os.listdir(checkpoint_dir) if "epoch" in x]}
                self.chkpoint = pjoin(checkpoint_dir, available_checkpoints[sorted(list(available_checkpoints.keys()))[-1]])
            else:
                self.chkpoint = pjoin(path2chk, "Checkpoints", "last.ckpt")
        else:
            self.chkpoint = None

        if hparams.taskID == 0 or hparams.taskID == 1:
            self.model = ReconEngine(**vars(hparams))
            if hparams.run_mode == 2 and bool(self.chkpoint):
                # TODO ckpt_path is not working during testing if not trained in the same run. So loading explicitly. check why
                print("Loading existing checkpoint...")
                self.model.load_state_dict(
                    torch.load(self.chkpoint)['state_dict'])
        loggers = []
        if hparams.wnbactive:
            loggers.append(WandbLogger(name=hparams.run_name, id=hparams.run_name, project=hparams.wnbproject,
                                       group=hparams.wnbgroup, entity=hparams.wnbentity, config=hparams))
            if bool(hparams.wnbmodellog) and hparams.wnbmodellog!= "None":
                loggers[-1].watch(self.model, log=hparams.wnbmodellog, log_freq=hparams.wnbmodelfreq)
        else:
            os.environ["WANDB_MODE"] = "dryrun"
        if hparams.tbactive:
            # TODO log_graph as True making it crash due to backward hooks
            os.makedirs(hparams.tblog_path, exist_ok=True)
            loggers.append(TensorBoardLogger(hparams.tblog_path,
                           name=hparams.run_name, log_graph=False))

        checkpoint_callback = ModelCheckpoint(
            dirpath=pjoin(hparams.save_path, hparams.run_name, "Checkpoints"),
            monitor='val_loss',
            save_last=True,
        )

        self.trainer = Trainer(
            logger=loggers,
            precision=16 if hparams.amp else 32,
            gpus=1,
            callbacks=[checkpoint_callback],
            checkpoint_callback=True,
            max_epochs=hparams.num_epochs,
            terminate_on_nan=True,
            deterministic=not hparams.non_deter,
            accumulate_grad_batches=hparams.accumulate_gradbatch,
            resume_from_checkpoint=self.chkpoint,
            check_val_every_n_epoch=1 if hparams.do_val else hparams.num_epochs+1,
            auto_scale_batch_size='binsearch' if hparams.auto_bs else None,
            auto_lr_find=hparams.auto_lr,
            log_every_n_steps=hparams.log_freq,
            flush_logs_every_n_steps=hparams.log_freq*2
        )

        if not hparams.non_deter:
            try: #As warn_only works with torch 1.11, this is for backward compatibility
                torch.use_deterministic_algorithms(True, warn_only=True) 
            except:
                torch.use_deterministic_algorithms(True)#, warn_only=True)

        self.hparams = hparams
        self.train_done = False

        # Impliest that only testing a "preweights" checkpoint
        if hparams.run_mode == 2 and bool(hparams.preweights_path):
            self.train_done = True

    def train(self):
        self.trainer.fit(self.model)
        self.train_done = True

    def test(self):
        os.makedirs(self.hparams.res_path, exist_ok=True)

        if self.train_done:
            if self.hparams.do_val:
                self.trainer.test(
                    test_dataloaders=self.model.test_dataloader())
            else:
                self.trainer.test(
                    model=self.model, test_dataloaders=self.model.test_dataloader())
        else:
            # , ckpt_path=self.chkpoint) #TODO: ckpt_path not working, check why
            self.trainer.test(model=self.model,
                              test_dataloaders=self.model.test_dataloader())

    def align(self):
        self.trainer.tune(self.model)

    def engage(self):
        if self.hparams.run_mode in [0, 1, 3, 4]:
            self.train()
        if self.hparams.run_mode in [2, 3, 4]:
            self.test()
