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
    parser.add_argument('--trainID', action="store", default="ResNet14_fullVol2D_L1Loss") ## "testing")  ## "ResNet14"
    parser.add_argument('--resume', action="store", default=0, type=int, help="To resume training from the last checkpoint") ## "testing")  ## "ResNet14"
    parser.add_argument('--load_best', action="store", default=1, type=int, help="To resume training from the last checkpoint") ## "testing")  ## "ResNet14"
    parser.add_argument('--gpu', action="store", default="0")
    parser.add_argument('--seed', action="store", default=1701, type=int)
    parser.add_argument('--num_workers', action="store", default=0, type=int)
    parser.add_argument('--batch_size', action="store", default=1, type=int)  
    parser.add_argument('--accumulate_gradbatch', action="store", default=1, type=int) ## 1 as default  
    # parser.add_argument('--datajson_path', action="store", default="executors/MoCo3D/datainfo_under_dummy.json")
    parser.add_argument('--datajson_path', action="store", default="executors/UnderRecon/datainfo_under.json")
    parser.add_argument('--tblog_path', action="store", default="/home/schatter/Soumick/Output/NCC1701/MoCo3D/TBLogs")
    parser.add_argument('--save_path', action="store", default="/home/schatter/Soumick/Output/NCC1701/MoCo3D/Results")
    parser.add_argument('--cuda', action="store_true", default=True)
    parser.add_argument('--amp', action="store_true", default=True)
    parser.add_argument('--run_mode', action="store", default=4, type=int, help='0: Train, 1: Train and Validate, 2:Test, 3: Train followed by Test, 4: Train and Validate followed by Test')
    parser.add_argument('--do_profile', action="store_true", default=False)
    parser.add_argument('--non_deter', action="store_true", default=False)
    parser.add_argument('--fftnorm', action="store", default="ortho")

    #Training params
    parser.add_argument('--num_epochs', action="store", default=2, type=int, help="Total number of epochs. If resuming, then it will continue till a total number of epochs set by this.")
    parser.add_argument('--lr', action="store", default=0.0001, type=float)
    parser.add_argument('--lossID', action="store", default=1, type=int, help="Loss ID."+str(LOSSID))
    parser.add_argument('--ploss_level', action="store", default=math.inf, type=int)
    parser.add_argument('--ploss_type', action="store", default="L1")
    parser.add_argument('--patch_size', action="store", default="", help="length, width, depth")
    parser.add_argument('--input_shape', action="store", default="256,256", help="length, width, depth (to be used if patch_size is not given)")
    parser.add_argument('--croppad', action="store_true", default=False, help="If True, then it will crop or pad the volume/slice to the given input_shape")  
    parser.add_argument('--patch_qlen', action="store", default=150, type=int)  ## 5000 - 50
    parser.add_argument('--patch_per_vol', action="store", default=150, type=int)  # 1000 - 10
    parser.add_argument('--patch_inference_strides', action="store", default="224,224,1", help="stride_length, stride_width, stride_depth")
    parser.add_argument('--im_log_freq', action="store", default=10, type=int, help="For Tensorboard image logs, n_iteration. Set it to -1 if not desired")
    parser.add_argument('--save_freq', action="store", default=1, type=int, help="For Checkpoint save, n_epochs")
    parser.add_argument('--save_inp', action="store", default=1, type=int, help="Whether to save the input during testing")
    parser.add_argument('--do_savenorm', action="store", default=1, type=int, help="Whether to normalise before saving and calculating metrics during testing")
    
    #Augmentations
    parser.add_argument('--p_contrast_augment', action="store", default=0.75, type=float, help="Probability of using contrast augmentations. Set it to 0 or -1 to avoid using.")
    parser.add_argument('--random_crop', action="store", default="", help="Randomly crop the given image, only ds_mode=1. Set it to None or blank if not to be used.")
    parser.add_argument('--p_random_crop', action="store", default=0.75, type=float, help="Probability of Randomcrop, only if is3D=False. This should be 1 if batch size is more than 1.")

    #Network Params
    parser.add_argument('--modelID', action="store", default=0, type=int, help="0: ReconResNet, 1: KSPReconResNet, 2: DualSpaceReconResNet")
    parser.add_argument('--preweights_path', action="store", default="", help="checkpoint path for pre-loading")
    parser.add_argument('--is3D', action="store", default=0, type=int, help="Is it a 3D model?")
    parser.add_argument('--model_dataspace_inp', action="store", default=0, type=int, help="Dataspace of the model's input. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--model_dataspace_gt', action="store", default=0, type=int, help="Dataspace of the model's groundturth. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--model_dataspace_out', action="store", default=0, type=int, help="Dataspace of the model's output. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--dataspace_out', action="store", default=0, type=int, help="Dataspace of the final output. 0: ImageSapce, 1: kSpace")
    parser.add_argument('--in_channels', action="store", default=1, type=int)
    parser.add_argument('--out_channels', action="store", default=1, type=int)
    parser.add_argument('--model_res_blocks', action="store", default=14, type=int, help="For ReconResNet")
    parser.add_argument('--model_starting_nfeatures', action="store", default=64, type=int, help="For ReconResNet, ShuffleUNet")
    parser.add_argument('--model_updown_blocks', action="store", default=2, type=int, help="For ReconResNet")
    parser.add_argument('--model_do_batchnorm', action="store_true", default=False, help="For ReconResNet")
    parser.add_argument('--model_relu_leaky', action="store_true", default=True, help="For ReconResNet")
    parser.add_argument('--model_is_replicatepad', action="store_true", default=False, help="For ReconResNet. Whether to use ReplacationPad instead of ReflectionPad, the later being the default")
    parser.add_argument('--model_forwardV', action="store", default=0, type=int, help="For ReconResNet")
    parser.add_argument('--model_drop_prob', action="store", default=0.2, type=float, help="For ReconResNet")
    parser.add_argument('--model_upinterp_algo', action="store", default="convtrans", help='"convtrans", or interpolation technique: "sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"')
    parser.add_argument('--model_out_act', action="store", default="sigmoid", help='For ReconResNet')
    parser.add_argument('--model_post_interp_convtrans', action="store_true", default=True, help="For ReconResNet")
    parser.add_argument('--model_dspace_connect_mode', action="store", default="serial", help='w_parallel, parallel, serial. For DualSpaceReconResNet')
    parser.add_argument('--model_inner_norm_ksp', action="store_true", default=True, help="For KSPReconResNet. DualSpaceReconResNet")
    
    parser.add_argument('--use_datacon', action="store_true", default=True, help="Use Data Consistency")

    parser.add_argument('--lr_decay_type', action="store", default=1, type=int, help='0: No Decay, 1: StepLR, 2: ReduceLROnPlateau')
    parser.add_argument('--lr_decay_nepoch', action="store", default=50, type=int, help='Decay the learning rate after every Nth epoch')
    parser.add_argument('--lr_decay_rate', action="store", default=0.1, type=float, help='Decay rate')

    #Model tunes with lightning
    parser.add_argument('--auto_bs', action="store", default=0, help="Automatically find the batch size to fit best")
    parser.add_argument('--auto_lr', action="store", default=0, help="Automatically find the LR")

    parser.add_argument('--ds_mode', action="store", default=1, type=int, help='0: TorchIO, 1: in-house MRITorchDS (medfile)')
    parser.add_argument('--ds2D_mid_n', action="store", default=-1, type=int, help='Number of mid slices to be used per volume. -1 for all. (Only for ds_mode=1 + is3D=False)')
    parser.add_argument('--ds2D_mid_per', action="store", default=-1, type=float, help='Percentage of mid slices to be used per volume, when mid_n is -1. -1 to ignore. (Only for ds_mode=1 + is3D=False)')
    parser.add_argument('--ds2D_random_n', action="store", default=-1, type=int, help='Number of random slices to be used per volume, when mid_n and mid_per are -1. -1 for all. (Only for ds_mode=1 + is3D=False)')
    parser.add_argument('--motion_mode', action="store", default=1, type=int, help='0: RandomMotionGhostingFast using TorchIO, 1: Motion2Dv0, 2: Motion2Dv1')

    #Motion parameters, for TorchIO RandomMotionGhosting or RandomMotionGhostingFast
    parser.add_argument('--motionmg_degrees', action="store", default="1.0,3.0", help="Tuple of Float, passed as CSV")
    parser.add_argument('--motionmg_translation', action="store", default="1.0,3.0", help="Tuple of Float, passed as CSV")
    parser.add_argument('--motionmg_num_transforms', action="store", default="2,10", help="Tuple of Int, passed as CSV")
    parser.add_argument('--motionmg_num_ghosts', action="store", default="2,5", help="Tuple of Int, passed as CSV")
    parser.add_argument('--motionmg_intensity', action="store", default="0.01,0.75", help="Tuple of Float, passed as CSV")
    parser.add_argument('--motionmg_restore', action="store", default="0.01,1.0", help="Tuple of Float, passed as CSV")
    parser.add_argument('--motionmg_image_interpolation', action="store", default='linear')
    parser.add_argument('--motionmg_ghosting_axes', action="store", default="0,1", help="Tuple of Int, passed as CSV")
    parser.add_argument('--motionmg_p_motion', action="store", default=0.75, type=float)
    parser.add_argument('--motionmg_p_ghosting', action="store", default=0.75, type=float)

    #Motion parameters, custom non-Torchio Motion corrupters
    parser.add_argument('--motion_p', action="store", type=float, default=0.8, help="Probability of the motion corrption being applied")
    parser.add_argument('--motion_sigma_range', action="store", default="1.0,3.0", help="Range of randomly-chosen sigma values. Tuple of Float, passed as CSV")
    parser.add_argument('--motion_n_threads', action="store", type=int, default=10, help="Number of threads to use")
    parser.add_argument('--motion_restore_original', action="store", type=float, default=0, help="Amount of original image to restore (Only for Motion2Dv1), set 0 to avoid")

    
    #TODO currently not in use, params are hardcoded 
    #Controlling motion corruption, whether to run on the fly or use the pre-created ones. If live_corrupt is True, only then the following params will be used
    # parser.add_argument('--corrupt_prob', action="store", default=0.75, type=float, help="Probability of the corruption to be applied or corrupted volume to be used")
    # parser.add_argument('--live_corrupt', action="store_true", default=False)
    # parser.add_argument('--motion_mode', action="store", default=2, type=int, help="Mode 0: TorchIO's, 1: Custom direction specific")
    # parser.add_argument('--motion_sigma', action="store", default=0.1, type=float, help="Only for motion_mode 2")
    # parser.add_argument('--motion_random_sigma', action="store_true", default=False, help="Only for motion_mode 2 - to randomise the sigma value, treating the provided sigma as upper limit and 0 as lower")
    # parser.add_argument('--motion_n_threads', action="store", default=8, type=int, help="Only for motion_mode 2 - to apply motion for each thread encoding line parallel, max thread controlled by this. Set to 0 to perform serially.")

    parser.add_argument("-tba", "--tbactive", type=int, default=0, help="User Tensorboard")

    #WnB related params
    parser.add_argument("-wnba", "--wnbactive", type=int, default=0, help="Use WandB")
    parser.add_argument("-wnbp", "--wnbproject", default='UnderRecon', help="WandB: Name of the project")
    parser.add_argument("-wnbe", "--wnbentity", default='soumick', help="WandB: Name of the entity")
    parser.add_argument("-wnbg", "--wnbgroup", default='NCC1701Set0', help="WandB: Name of the group")
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
