import argparse
import math
import os
import sys
sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the executors

import torch

from Bridge.MainEngine import Engine
from Engineering.constants import *
from pytorch_lightning import seed_everything

seed_everything(1701)

def getARGSParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskID', action="store", type=int, default=0, help="0: Undersampled Recon, 1: MoCo, 2: Classification") ## "testing")  ## "ResNet14"
    # parser.add_argument('--trainID', action="store", default="testing") ## "testing")  ## "ResNet14"
    parser.add_argument('--trainID', action="store", default="reproAttempt8new_100ep-IXIT1HHVarden1D15") ## "testing")  ## "ResNet14"
    parser.add_argument('--resume', action="store", default=0, type=int, help="To resume training from the last checkpoint") ## "testing")  ## "ResNet14"
    parser.add_argument('--load_best', action="store", default=1, type=int, help="To resume training from the last checkpoint") ## "testing")  ## "ResNet14"
    parser.add_argument('--gpu', action="store", default="0")
    parser.add_argument('--seed', action="store", default=1701, type=int)
    parser.add_argument('--num_workers', action="store", default=0, type=int)
    parser.add_argument('--batch_size', action="store", default=1, type=int) ## 256    
    parser.add_argument('--accumulate_gradbatch', action="store", default=1, type=int) ## 256    
    # parser.add_argument('--datajson_path', action="store", default="executors/MoCo3D/datainfo_under_dummy.json")
    parser.add_argument('--datajson_path', action="store", default="executors/UnderRecon/datainfo_under.json")
    parser.add_argument('--tblog_path', action="store", default="/project/schatter/NCC1701repro/TBLogs")
    parser.add_argument('--save_path', action="store", default="/project/schatter/NCC1701repro/Results")
    parser.add_argument('--cuda', action="store_true", default=True)
    parser.add_argument('--amp', action="store_true", default=False)
    parser.add_argument('--run_mode', action="store", default=4, type=int, help='0: Train, 1: Train and Validate, 2:Test, 3: Train followed by Test, 4: Train and Validate followed by Test')
    parser.add_argument('--do_profile', action="store_true", default=False)
    parser.add_argument('--non_deter', action="store_true", default=False)
    parser.add_argument('--fftnorm', action="store", default="ortho")

    #Training params
    parser.add_argument('--num_epochs', action="store", default=100, type=int, help="Total number of epochs. If resuming, then it will continue till a total number of epochs set by this.")
    parser.add_argument('--lr', action="store", default=0.0001, type=float)
    parser.add_argument('--lossID', action="store", default=3, type=int, help="Loss ID."+str(LOSSID))
    parser.add_argument('--ploss_level', action="store", default=math.inf, type=int)
    parser.add_argument('--ploss_type', action="store", default="L1")
    parser.add_argument('--patch_size', action="store", default="256,256,1", help="length, width, depth")
    parser.add_argument('--input_shape', action="store", default="", help="length, width, depth (to be used if patch_size is not given)")
    parser.add_argument('--patch_qlen', action="store", default=150, type=int)  ## 5000 - 50
    parser.add_argument('--patch_per_vol', action="store", default=150, type=int)  # 1000 - 10
    parser.add_argument('--patch_inference_strides', action="store", default="256,256,1", help="stride_length, stride_width, stride_depth")
    parser.add_argument('--im_log_freq', action="store", default=10, type=int, help="For Tensorboard image logs, n_iteration. Set it to -1 if not desired")
    parser.add_argument('--save_freq', action="store", default=1, type=int, help="For Checkpoint save, n_epochs")
    parser.add_argument('--save_inp', action="store", default=1, type=int, help="Whether to save the input during testing")
    parser.add_argument('--do_savenorm', action="store", default=1, type=int, help="Whether to normalise before saving and calculating metrics during testing")
    
    #Augmentations
    parser.add_argument('--contrast_augment', action="store", default=0, type=int, help="Whether to user contrast augmentations")

    #Network Params
    parser.add_argument('--modelID', action="store", default=0, type=int, help="0: RecoNResNet, 1: ResNetHHPaper")
    parser.add_argument('--preweights_path', action="store", default="", help="checkpoint path for pre-loading")
    parser.add_argument('--is3D', action="store", default=0, type=int, help="Is it a 3D model?")
    parser.add_argument('--model_dataspace_inp', action="store", default=0, type=int, help="Dataspace of the model's input. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--model_dataspace_gt', action="store", default=0, type=int, help="Dataspace of the model's groundturth. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--model_dataspace_out', action="store", default=0, type=int, help="Dataspace of the model's output. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--dataspace_out', action="store", default=0, type=int, help="Dataspace of the final output. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--in_channels', action="store", default=1, type=int)
    parser.add_argument('--out_channels', action="store", default=1, type=int)
    parser.add_argument('--model_res_blocks', action="store", default=14, type=int, help="For RecoNResNet")
    parser.add_argument('--model_starting_nfeatures', action="store", default=64, type=int, help="For RecoNResNet, ShuffleUNet")
    parser.add_argument('--model_updown_blocks', action="store", default=2, type=int, help="For RecoNResNet")
    parser.add_argument('--model_do_batchnorm', action="store_true", default=False, help="For RecoNResNet")
    parser.add_argument('--model_relu_leaky', action="store_true", default=True, help="For RecoNResNet")
    parser.add_argument('--model_forwardV', action="store", default=0, type=int, help="For RecoNResNet")
    parser.add_argument('--model_drop_prob', action="store", default=0.2, type=float, help="For RecoNResNet")
    parser.add_argument('--model_upinterp_algo', action="store", default="convtrans", help='"convtrans", or interpolation technique: "sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"')
    parser.add_argument('--model_post_interp_convtrans', action="store_true", default=False, help="For RecoNResNet")
    
    parser.add_argument('--use_datacon', action="store_true", default=True, help="Use Data Consistency")

    parser.add_argument('--lr_decay_type', action="store", default=1, type=int, help='0: No Decay, 1: StepLR, 2: ReduceLROnPlateau')
    parser.add_argument('--lr_decay_nepoch', action="store", default=50, type=int, help='Decay the learning rate after every Nth epoch')
    parser.add_argument('--lr_decay_rate', action="store", default=0.1, type=float, help='Decay rate')

    #Model tunes with lightning
    parser.add_argument('--auto_bs', action="store", default=0, help="Automatically find the batch size to fit best")
    parser.add_argument('--auto_lr', action="store", default=0, help="Automatically find the LR")

    #TODO currently not in use, params are hardcoded 
    #Controlling motion corruption, whether to run on the fly or use the pre-created ones. If live_corrupt is True, only then the following params will be used
    parser.add_argument('--corrupt_prob', action="store", default=0.75, type=float, help="Probability of the corruption to be applied or corrupted volume to be used")
    parser.add_argument('--live_corrupt', action="store_true", default=False)
    parser.add_argument('--motion_mode', action="store", default=2, type=int, help="Mode 0: TorchIO's, 1: Custom direction specific")
    parser.add_argument('--motion_degrees', action="store", default=10, type=int)
    parser.add_argument('--motion_translation', action="store", default=10, type=int)
    parser.add_argument('--motion_num_transforms', action="store", default=10, type=int)
    parser.add_argument('--motion_image_interpolation', action="store", default='linear')
    parser.add_argument('--motion_norm_mode', action="store", default=2, type=int, help="norm_mode 0: No Norm, 1: Divide by Max, 2: MinMax")
    parser.add_argument('--motion_noise_dir', action="store", default=1, type=int, help="noise_dir 0 1 or 2 (only for motion_mode 1, custom direction specific). noise_dir=2 will act as motion_mode 0. noise_dir=-1 will randomly choose 0 or 1.")
    parser.add_argument('--motion_mu', action="store", default=0.0, type=float, help="Only for motion_mode 2")
    parser.add_argument('--motion_sigma', action="store", default=0.1, type=float, help="Only for motion_mode 2")
    parser.add_argument('--motion_random_sigma', action="store_true", default=False, help="Only for motion_mode 2 - to randomise the sigma value, treating the provided sigma as upper limit and 0 as lower")
    parser.add_argument('--motion_n_threads', action="store", default=8, type=int, help="Only for motion_mode 2 - to apply motion for each thread encoding line parallel, max thread controlled by this. Set to 0 to perform serially.")

    parser.add_argument("-tba", "--tbactive", type=int, default=1, help="User Tensorboard")

    #WnB related params
    parser.add_argument("-wnba", "--wnbactive", type=int, default=1, help="Use WandB")
    parser.add_argument("-wnbp", "--wnbproject", default='reproNCC1701', help="WandB: Name of the project")
    parser.add_argument("-wnbe", "--wnbentity", default='soumick', help="WandB: Name of the entity")
    parser.add_argument("-wnbg", "--wnbgroup", default='RecoNResNet', help="WandB: Name of the group")
    parser.add_argument("-wnbpf", "--wnbprefix", default='', help="WandB: Prefix for TrainID")
    parser.add_argument("-wnbml", "--wnbmodellog", default='all', help="WandB: While watching the model, what to save: gradients, parameters, all, None")
    parser.add_argument("-wnbmf", "--wnbmodelfreq", type=int, default=100, help="WandB: The number of steps between logging gradients")
    
    return parser

if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = getARGSParser()
    engine = Engine(parser)
    if engine.hparams.auto_bs or engine.hparams.auto_lr:
        print("Engine alignment initiating..")
        engine.align()
        print("Engine alignment finished.")
    engine.engage()