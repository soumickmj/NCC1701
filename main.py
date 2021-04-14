import random
import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np
import socket
import importlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pytorch_ssim
from utils.HandleTorchModel import GetModelMemoryUsage
from model.Helper import Helper
from utils.GetFreeGPU import get_free_gpu
from utils.TorchLoss.ImgSSIMFromFourierLoss import ImgSSIMFromFourierLoss
import pytorch_ssim
from utils.fastMRI.subsample import MaskFunc
from utils.TorchLoss import MultiSSIM
from utils.TorchLoss.WeightedLoss import GaussianWeightedLoss 

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Change is the only constant. Remote debug isn't working yet"

def parseARGS():
    parser = argparse.ArgumentParser(description='The Bridge of the NCC1701')
    parser.add_argument('--trainID', action="store", default="XAttempt0-fastMRIBrain-MotionRotLim5-Resnet2Dv2b1413-SSIMImgLoss-AllSlices", help='Train ID will be used to create the folder to save output')
    parser.add_argument('--gpu', action="store", default="0", help='GPU ID, For multiple GPU, use coma seperated')
    parser.add_argument('--multi_gpu', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--cpu_only', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--no_amp', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--seed', action="store", default=1701, type=int, help='TODO')
    parser.add_argument('--run_mode', action="store", default=2, type=int, help='0: Train 1: Train and Validate 2:Test 3: Train followed by Test, 4: Train and Validate followed by Test')
    parser.add_argument('--non_deter', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--domain', action="store", default="image", help='In which domain, network should work? Acceptable values: | fourier | hartley | image | compleximage |')
    parser.add_argument('--n_coils', action="store", default=1, type=int, help='TODO')
    parser.add_argument('--filter_train', action="store", nargs="+", default="Skyra", help="['Aera','Biograph_mMR','Prisma_fit', 'Skyra']")
    parser.add_argument('--filter_val', action="store", nargs="+", default="Skyra", help="['Aera','Biograph_mMR','Prisma_fit', 'Skyra']")
    parser.add_argument('--filter_test', action="store", nargs="+", default="Skyra", help="['Aera','Biograph_mMR','Prisma_fit', 'Skyra']")
    parser.add_argument('--corupt_type', action="store", default="fastmriunder", help='TODO')
    parser.add_argument('--center_fractions', action="store", nargs="+", default=0.08, type=float, help="TODO: Only fastMRI")
    parser.add_argument('--acceleration_factors', action="store", nargs="+", default=4, type=int, help="TODO: Only fastMRI")
    parser.add_argument('--mask_seed', action="store", default=1701, type=int, help='Only fastMRI: Either a number, array of numbers or None to keep it complete random')
    parser.add_argument('--top_n_train', action="store", nargs="+", default=50, type=int, help="TODO")
    parser.add_argument('--top_n_val', action="store", nargs="+", default=10, type=int, help="TODO")
    parser.add_argument('--top_n_test', action="store", nargs="+", default=10, type=int, help="TODO")
    parser.add_argument('--undersamp_mat', action="store", default=None, help='path to a MAT file containing undersampling mask. TODO')
    parser.add_argument('--model_type', action="store", default="2DSingleSlice", help='Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice')
    parser.add_argument('--model_type_actual', action="store", default="2DMultiSlice", help='Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice')
    parser.add_argument('--is_radial', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--not_save_best', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--save_frequency', action="store", default=1, type=int, help='TODO')
    parser.add_argument('--log_frequency', action="store", default=10, type=int, help='TODO')
    parser.add_argument('--resume', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--checkpoint_path', action="store", default="fastmriunder", help='TODO')
    parser.add_argument('--dataset', action="store", default="utils.TorchDS.fastMRI", help='TODO')
    parser.add_argument('--net', action="store", default="model.kSpace.CUNet", help='TODO')
    parser.add_argument('--net_type', action="store", default=0, type=int, help='0: Oridinary, 1: KSPNet, 2: SuperResNet')
    parser.add_argument('--engine', action="store", default="model.MainEngine", help='TODO')
    parser.add_argument('--batch_size', action="store", default=1, type=int, help='TODO')
    parser.add_argument('--image_size', action="store", default=None, type=int, help='None to use the whole image, single int to crop square. TODO: rectanglular. Not implimented')
    parser.add_argument('--n_channels', action="store", default=1, type=int, help='TODO')
    parser.add_argument('--n_workers_train', action="store", default=0, type=int, help='TODO')
    parser.add_argument('--n_workers_test', action="store", default=0, type=int, help='TODO')
    parser.add_argument('--n_epoch', action="store", default=100, type=int, help='TODO')
    parser.add_argument('--valid_percent', action="store", default=0.2, type=float, help='Percentage of the complete dataset to be used for creating the validation set.')
    parser.add_argument('--pin_memory', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--loss_type', action="store", default=0, type=int, help='0: Calculate as it is, 1: bring to image space before loss calculation, 2: same as 1 but before loss, normalize')
    parser.add_argument('--neg_loss', action="store_true", default=False, type=bool, help='TODO')
    parser.add_argument('--init_lr', action="store", default=0.0001, type=float, help='TODO')
    parser.add_argument('--lr_decay_type', action="store", default=0, type=int, help='0: No Decay, 1: StepLR, 2: ReduceLROnPlateau')
    parser.add_argument('--lr_decay_nepoch', action="store", default=25, type=int, help='Decay the learning rate after every Nth epoch')
    parser.add_argument('--lr_decay_rate', action="store", default=0.1, type=float, help='Decay rate')
    parser.add_argument('--beta1', action="store", default=0.9, type=float, help='Momentum parameter of the gradients')
    parser.add_argument('--beta2', action="store", default=0.999, type=float, help='Momentum parameter of the gradients squared ')
    parser.add_argument('--epsilon', action="store", default=1e-09, type=float, help='Small float added to avoid division by zero')

    pc_name = socket.gethostname()
    if(pc_name == 'BMMR-Soumick'): ##For BMMR-Soumick
        parser.add_argument('--tb_logroot', action="store", default=r'D:\CloudData\OneDrive\OvGU\Tensorboard_Logs\Ent-Gen3', help='Root for Tensorboard logs')
        parser.add_argument('--train_path_fully', action="store", default=r'B:\Soumick\Challange DSs\fastMRI\Brain\multicoil_train', help='Root path to Train DS')
        parser.add_argument('--test_path_fully', action="store", default=r'B:\Soumick\Challange DSs\fastMRI\Brain\multicoil_train', help='Root path to Test DS')
        parser.add_argument('--val_path_fully', action="store", default=r'B:\Soumick\Challange DSs\fastMRI\Brain\multicoil_train', help='Root path to Val DS')
        parser.add_argument('--ds_split_xlsx_train', action="store", default=r"B:\Soumick\Challange DSs\fastMRI\Brain\Splits\train_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Train Split')
        parser.add_argument('--ds_split_xlsx_test', action="store", default=r"B:\Soumick\Challange DSs\fastMRI\Brain\Splits\test_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Test Split')
        parser.add_argument('--ds_split_xlsx_val', action="store", default=r"B:\Soumick\Challange DSs\fastMRI\Brain\Splits\val_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Val Split')
        parser.add_argument('--output_path', action="store", default=r'D:\Output\Gen3\fastMRI\Brain\AllSlices', help='Root path to store output')
    elif(pc_name == 'brain'): ##For AleMaxi's Brain
        parser.add_argument('--tb_logroot', action="store", default=r'/home/duennwal/soumick/TBLogs', help='Root for Tensorboard logs')
        parser.add_argument('--train_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Train DS')
        parser.add_argument('--test_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Test DS')
        parser.add_argument('--val_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Val DS')
        parser.add_argument('--ds_split_xlsx_train', action="store", default=r"/pool/public/data/fastMRI/train_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Train Split')
        parser.add_argument('--ds_split_xlsx_test', action="store", default=r"/pool/public/data/fastMRI/test_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Test Split')
        parser.add_argument('--ds_split_xlsx_val', action="store", default=r"/pool/public/data/fastMRI/val_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Val Split')
        parser.add_argument('--output_path', action="store", default=r'/home/duennwal/soumick/Output', help='Root path to store output')
    elif(pc_name == "n00-01"): ##For FIN FCM Cluster: node00 kino 01 - TODO
        parser.add_argument('--tb_logroot', action="store", default=r'/scratch/tmp/schatter/Output/Neural/TensorBoardLogs/Ent-Gen3', help='Root for Tensorboard logs')
        parser.add_argument('--train_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Train DS')
        parser.add_argument('--test_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Test DS')
        parser.add_argument('--val_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Val DS')
        parser.add_argument('--ds_split_xlsx_train', action="store", default=r"/pool/public/data/fastMRI/train_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Train Split')
        parser.add_argument('--ds_split_xlsx_test', action="store", default=r"/pool/public/data/fastMRI/test_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Test Split')
        parser.add_argument('--ds_split_xlsx_val', action="store", default=r"/pool/public/data/fastMRI/val_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Val Split')
        parser.add_argument('--output_path', action="store", default=r'/home/duennwal/soumick/Output', help='Root path to store output')
    elif(pc_name == "u145-13"): ##For FIN DBMS V100 Server - TODO
        parser.add_argument('--tb_logroot', action="store", default=r'/scratch/tmp/schatter/Output/Neural/TensorBoardLogs/Ent-Gen3', help='Root for Tensorboard logs')
        parser.add_argument('--train_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Train DS')
        parser.add_argument('--test_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Test DS')
        parser.add_argument('--val_path_fully', action="store", default=r'/pool/public/data/fastMRI/multicoil_train', help='Root path to Val DS')
        parser.add_argument('--ds_split_xlsx_train', action="store", default=r"/pool/public/data/fastMRI/train_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Train Split')
        parser.add_argument('--ds_split_xlsx_test', action="store", default=r"/pool/public/data/fastMRI/test_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Test Split')
        parser.add_argument('--ds_split_xlsx_val', action="store", default=r"/pool/public/data/fastMRI/val_AXT2_16Coil_768x396_Split503020.xlsx", help='Excel file containning the Val Split')
        parser.add_argument('--output_path', action="store", default=r'/home/duennwal/soumick/Output', help='Root path to store output')
    elif(pc_name == 'gpu18.urz.uni-magdeburg.de'): #For GPU18 Server
        parser.add_argument('--tb_logroot', action="store", default=r'/nfs1/schatter/TBLogs/Enterprise/v2-Gen4', help='Root for Tensorboard logs')
        parser.add_argument('--train_path_fully', action="store", default=r'/nfs1/schatter/fastMRI_Brain/multicoil_train', help='Root path to Train DS')
        parser.add_argument('--test_path_fully', action="store", default=r'/nfs1/schatter/fastMRI_Brain/multicoil_train', help='Root path to Test DS')
        parser.add_argument('--val_path_fully', action="store", default=r'/nfs1/schatter/fastMRI_Brain/multicoil_train', help='Root path to Val DS')
        parser.add_argument('--ds_split_xlsx_train', action="store", default=r"/nfs1/schatter/fastMRI_Brain/Splits/train_MultiContrast_16Coil_640x320_Split503020.xlsx", help='Excel file containning the Train Split')
        parser.add_argument('--ds_split_xlsx_test', action="store", default=r"/nfs1/schatter/fastMRI_Brain/Splits/test_MultiContrast_16Coil_640x320_Split503020.xlsx", help='Excel file containning the Test Split')
        parser.add_argument('--ds_split_xlsx_val', action="store", default=r"/nfs1/schatter/fastMRI_Brain/Splits/val_MultiContrast_16Coil_640x320_Split503020.xlsx", help='Excel file containning the Val Split')
        parser.add_argument('--output_path', action="store", default=r'/nfs1/schatter/Output/Enterprise/v2-Gen4/fasMRI/Brain/MoCo', help='Root path to store output')
    else:
        sys.exit('PC Name Recognized. So, path definations not initialized')


    return parser.parse_args()

if __name__ == "__main__":
    args = parseARGS()
    IsCuda = True if torch.cuda.is_available() and not args.cpu_only else False
    device = torch.device("cuda:0" if IsCuda else "cpu")
    torch.backends.cudnn.benchmark = args.non_deter
    if not args.non_deter:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if(args.domain == 'fourier' or args.domain == 'compleximage'): #TODO: for complex hartley
        n_channels = args.n_channels*args.n_coils*2
    else:
        n_channels = args.n_channels*args.n_coils

    dsClass = importlib.import_module(args.dataset).MRITorchDS
    netModel = importlib.import_module(args.model).CUNet 
    Model = importlib.import_module(args.model_handler).Engine

    loss_func = pytorch_ssim.SSIM() #TODO: make config
    optimizer = optim.Adam #TODO: make config 

    if args.resume: #If IsResume is set to true, then resume from a given path or ask for a checkpoint path
        if args.checkpoint_path is '':
            root = tk.Tk()
            root.withdraw()
            CheckpointPath = filedialog.askopenfilename()
        else:
            CheckpointPath = args.checkpoint_path
        checkpoint = torch.load(CheckpointPath, map_location=device)
        last_epoch = checkpoint['epoch']

    if args.lr_decay_type == 1:
        lrScheduler_func = optim.lr_scheduler.StepLR
        lrScheduler_param_dict = {"step_size": args.lr_decay_nepoch,
                                    "gamma": args.lr_decay_rate, 
                                    "last_epoch": last_epoch if args.resume else -1} 
    elif args.lr_decay_type == 2:
        lrScheduler_func = optim.lr_scheduler.ReduceLROnPlateau
        lrScheduler_param_dict = {"factor": args.lr_decay_rate, 
                                    "mode": "min", #TODO: args. Min for decreasing loss, max for increasing
                                    "patience": 10, #TODO: args. Number of epochs with no improvement after which learning rate will be reduced.
                                    "threshold": 1e-4, #TODO: args. Threshold for measuring the new optimum, to only focus on significant changes.
                                    "cooldown": 0, #TODO: args. Number of epochs to wait before resuming normal operation after lr has been reduced.
                                    "min_lr": 0} #TODO: args. A lower bound on the learning rate of all param groups or each group respectively.
    else:
        lrScheduler_func = None
        lrScheduler_param_dict = None

    undersamplingMask = args.undersamp_mat #TODO: read Mask from the given mat file
    
    output_path = os.path.join(args.output_path, args.trainID)
    tb_logpath = os.path.join(args.tb_logroot, args.trainID)

    model = Model(args.domain, args.batch_size, imageSize=args.image_size, type = args.net_type, pin_memory = args.pin_memory, IsRadial=args.is_radial, 
                    undersamplingMask=undersamplingMask, log_path=tb_logpath, log_freq=args.log_frequency, use_amp=not args.no_amp)
    model.CreateModel(netModel, device, IsCuda=IsCuda, IsMultiGPU=args.multi_gpu, n_channels=n_channels, 
                      initialLearningRate=args.init_lr, lrScheduler_func=lrScheduler_func, lrScheduler_param_dict=lrScheduler_param_dict, 
                      betas=(args.beta1, args.beta2), epsilon=args.epsilon, 
                      loss_func=loss_func, optimizer=optimizer, NetType=args.net_type, IsNegetiveLoss=args.neg_loss, loss_type=args.loss_type)
    if args.resume:
        model, start_epoch = model.helpOBJ.load_checkpoint(checkpoint=checkpoint, model=model)
        if model.net.device != device:
            model.toDevice(device, args.multi_gpu)
    else:
        start_epoch = 0

    #Run according to the mentioned run mode
    if(args.run_mode == 0):
        train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
        model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path, save_frequency=save_frequency)
    elif(args.run_mode==1):
        if val_path_fully is None: 
            _, train_setDS = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train) 
            model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)
        else:
            print('Val set also supplied')
            train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
            val_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, val_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_val, ds_split_xlsx_val, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_val)
            model.TrainNValidate(train_loader=train_set, valid_loader=val_set, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)        
    elif(args.run_mode==2 and args.resume == True):
        test_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, test_path_fully, None, None, n_workers_test, getROIRes, undersampling_mat, filename_filter_test, ds_split_xlsx_test, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_test)
        model.Test(dataloader=test_set, output_path=output_path)
    elif(args.run_mode==3):
        train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
        model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path, save_frequency=save_frequency)

        test_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, test_path_fully, None, None, n_workers_test, getROIRes, undersampling_mat, filename_filter_test, ds_split_xlsx_test, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_test)
        model.Test(dataloader=test_set, output_path=output_path, sliceno = slicenoTest)
    elif(args.run_mode==4):
        if val_path_fully is None: 
            _, train_setDS = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train) 
            model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)
        else:
            print('Val set also supplied')
            train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
            val_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, val_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_val, ds_split_xlsx_val, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_val)
            model.TrainNValidate(train_loader=train_set, valid_loader=val_set, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)
            
        test_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, test_path_fully, None, None, n_workers_test, getROIRes, undersampling_mat, filename_filter_test, ds_split_xlsx_test, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_test)
        model.Test(dataloader=test_set, output_path=output_path)
    else:
        print('Invalid Run Mode. Remember: Test mode requires IsResume set to true')

    

    getROIRes = None #Can be a tuple like (320,320), then a cropping will be performed in the image space. TODO: not yet implimented
    mask_func = MaskFunc(center_fractions, acceleration_factors)


    os.environ["CUDA_VISIBLE_DEVICES"]=gpuID


