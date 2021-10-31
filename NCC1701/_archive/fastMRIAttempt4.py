#!/usr/bin/env python

"""
This is the main module, entry point of our program.
Settings of parameters are implimented here.
Hyperparameter updates can/should be done here

For reference regarding remote debugging, visit: https://docs.microsoft.com/en-us/visualstudio/python/debugging-python-code-on-remote-linux-machines
for VSCode support, https://code.visualstudio.com/docs/python/python-tutorial and https://donjayamanne.github.io/pythonVSCodeDocs/docs/debugging_remote-debugging/

This code was originally written for Undersampled images, so all naming conventions are according to that.
For Contrast Translation,
Fully: Destination Contrast
Under: Source Contrast
For Example,
Fully: T2, Under: T1 - For Converting T1 to T2

"""

###Resnet34	2DSingleSlice	No	200	mario		Alex3	varden20


import os
import sys
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

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Change is the only constant. Remote debug isn't working yet"

#################
##For remote debugging
#import ptvsd 

#ptvsd.enable_attach('dcgan1_torch_mick') #dcgan1_torch_mick is the remote debug service name in the remote server
#ptvsd.wait_for_attach()
#ptvsd.break_into_debugger()
#################

#################
##For initialization parameters

torch.autograd.set_detect_anomaly(True)

neumann_name = "c502.vc-a"

gpuID = "0"
#gpuID = "0,1,2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"]=gpuID
deterministic_model = True

#### Basic parameters
domain = 'fourier' #In which domain, network should work? Acceptable values: | fourier | hartley | image | compleximage |
n_coils = 15
if(domain == 'fourier' or domain == 'compleximage'):
    n_channels = n_coils*2
else:
    n_channels = n_coils
getROIRes = None #Can be a tuple like (320,320), then a cropping will be performed in the image space. TODO: not yet implimented
filename_filter_train = ['Skyra'] #['Aera','Biograph_mMR','Prisma_fit', 'Skyra']
filename_filter_test = ['Skyra'] #['Aera','Biograph_mMR','Prisma_fit', 'Skyra']
filename_filter_val = ['Skyra'] #['Aera','Biograph_mMR','Prisma_fit', 'Skyra']
corupt_type='fastmriunder'
center_fractions = [0.08]
acceleration_factors = [4]
mask_func = MaskFunc(center_fractions, acceleration_factors)
mask_seed=1701 #Either a number, array of numbers or None to keep it complete random
top_n_sampled_train = 50
top_n_sampled_val = 10
top_n_sampled_test = 10

undersampling_mat=None #TODO: DO THEM

modelType = '2DSingleSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
modelType_actual = '2DMultiSlice'
IsRadial = False

run_mode = 2 #0: Train 1: Train and Validate 2:Test 3: Train followed by Test, 4: Train and Validate followed by Test
save_best = True
save_frequency = 1
IsResume = True
CheckpointPath = r"D:\Output\Gen3\fastMRI\Skyra\AllSlices\Attempt4-fastMRISkyra-AccFact4Center08-CUNetResNetv214-MSELossImg-AllSlices!!!!\checkpoints\checkpoint.pth.tar"

#### CUDA Related Parameters
IsCuda = True
cudnn.benchmark = not deterministic_model
IsMultiGPU = False

if deterministic_model:
    torch.manual_seed(1701)
    cudnn.deterministic = True


#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.MATHandles.MAT').MRIMATTorchDS
dsClass = importlib.import_module('utils.TorchDS.fastMRI').MRITorchDS

#### Net Model Import path or object of the net model itself. 
netModel = importlib.import_module('model.kSpace.EnsambleSpace').EnsambleSpace 
NetType = 1 #0: Oridinary, 1: KSPNet, 2: SuperResNet

#### Load the custom model class
Model = importlib.import_module('model.ModelFastMRI').Model 

##### Parameters related to training, testing and validation
batch_size = 1
n_workers_train = 0
n_workers_test = 0
n_epoch = 100
valid_percent=0.20 #Percentage of the complete dataset to be used for creating the validation set.
pinMemory = False
loss_func = nn.MSELoss(reduction='mean')
#loss_func = nn.KLDivLoss(reduction='batchmean')
#loss_func = MultiSSIM.MultiSSIM
IsNegetiveLoss = False
optimizer=optim.Adam

#### Parameters related to the optimizer.

initialLearningRate = 0.0001#0.0002 #0.0001 #0.1
lrDecayNEpoch = 25 #4 #Decay the learning rate after every 4th epoch
lrDecayRate = 0.1 #Decay divide by 10 #0.5 #Decay by half
beta1 = 0.9 #0.5 #0.9 #Momentum parameter of the gradients
beta2 = 0.999 #Momentum parameter of the gradients squared 
epsilon = 1e-09 #Small float added to avoid division by zero


#### Define path to dataset and also to output - based on machine name
pc_name = socket.gethostname()
if(pc_name == 'BMMR-Soumick'):
    ##For BMMR-Soumick - For Undersampled Images. Fot MAT DSs, only the path given for fully is utilized
    tb_logroot = r'D:\CloudData\OneDrive\OvGU\Tensorboard_Logs\Ent-Gen3'
    train_path_fully = r'E:\Datasets\FastMRI\raw\multicoil_train\multicoil_train'
    test_path_fully = r'E:\Datasets\FastMRI\raw\multicoil_val\multicoil_val' #TODO: Hotfix done, using val set in place of test as test doesn't contain fully sampled kspace
    val_path_fully = r'E:\Datasets\FastMRI\raw\multicoil_val\multicoil_val'
    ds_split_xlsx_train = r"E:\Datasets\FastMRI\raw\multicoil_train\headers.xlsx"
    ds_split_xlsx_test = r"E:\Datasets\FastMRI\raw\multicoil_val\headers.xlsx"
    ds_split_xlsx_val = r"E:\Datasets\FastMRI\raw\multicoil_val\headers.xlsx"
    output_path = r'D:\Output\Gen3\fastMRI\Skyra\AllSlices'
    output_folder = 'Attempt4-fastMRISkyra-AccFact4Center08-CUNetResNetv214-MSELossImg-AllSlices-35ep'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
elif(pc_name == neumann_name):
    ##For Neumann - For Undersampled Images
    tb_logroot = '/scratch/tmp/schatter/Output/Neural/TensorBoardLogs\Ent-Gen3'
    train_path_fully = r'E:\Datasets\IXI\MATs\IXI-T1\Guys\Slice60to90\Varden1D30SameMask\train'
    test_path_fully = r'E:\Datasets\IXI\MATs\IXI-T1\Guys\Slice60to90\Varden1D30SameMask\test'
    val_path_fully = r'E:\Datasets\IXI\MATs\IXI-T1\Guys\Slice60to90\Varden1D30SameMask\val'
    ds_split_xlsx_train = None
    ds_split_xlsx_test = None
    ds_split_xlsx_val = None
    output_path = r'D:\Output\Gen3\IXI-T1\Guys\Slice60to90'
    output_folder = 'Attempt1-IXIT1GuysVarden1D30-Resnet2Dv214PReLU-SSIMLoss-Slice60to90'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
elif(pc_name == 'gpu18.urz.uni-magdeburg.de'):
    ##For GPU18 Cluster - For Undersampled Images
    tb_logroot = '/raid/scratch/schatter/Output/TensorBoardLogs/Ent-Gen3'
    train_path_fully = r'/nfs1/schatter/IXI/MATs/IXI-T1/HH/AllSlices/1DVarden30Mask/train'
    test_path_fully = r'/nfs1/schatter/IXI/MATs/IXI-T1/HH/AllSlices/1DVarden30Mask/test'
    val_path_fully = r'/nfs1/schatter/IXI/MATs/IXI-T1/HH/AllSlices/1DVarden30Mask/val'    
    ds_split_xlsx_train = None
    ds_split_xlsx_test = None
    ds_split_xlsx_val = None
    output_path = r'/raid/scratch/schatter/Output/Enterprise/v1-Gen3/IXI-T1/HH/AllSlices'
    output_folder = 'Attempt7-IXIT1HHVarden1D30-Resnet2Dv214PReLU-SSIMLoss-AllSlices'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
else:
    sys.exit('PC Name Recognized. So, path definations not initialized')

######################


######################
##Main heart of the code

model = Model(domain, batch_size, type = modelType, pin_memory = pinMemory, log_path=tb_logpath)
model.CreateModel(netModel, IsCuda = IsCuda, IsMultiGPU = IsMultiGPU, n_channels=n_channels, 
                      initialLearningRate=initialLearningRate, lrDecayNEpoch=lrDecayNEpoch, lrDecayRate=lrDecayRate, betas=(beta1, beta2), epsilon=epsilon, 
                      loss_func=loss_func, optimizer=optimizer, NetType = NetType, IsNegetiveLoss=IsNegetiveLoss)
if(IsResume): #If IsResume is set to true, then resume from a given path or ask for a checkpoint path
    if(CheckpointPath is ''):
        root = tk.Tk()
        root.withdraw()
        CheckpointPath = filedialog.askopenfilename()
    model, start_epoch = model.helpOBJ.load_checkpoint(checkpoint_file = CheckpointPath, model=model)
    if model.IsCuda != IsCuda: #If IsCuda status changed from last time
        if(IsCuda):
            model.ToCUDA(IsMultiGPU)
        else:
            model.ToCPU()
else:   
    start_epoch = 0

#Run according to the mentioned run mode
if(run_mode == 0):
    train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path, save_frequency=save_frequency)
elif(run_mode==1):
    if val_path_fully is None: 
        _, train_setDS = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train) 
        model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)
    else:
        print('Val set also supplied')
        train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
        val_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, val_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_val, ds_split_xlsx_val, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_val)
        model.TrainNValidate(train_loader=train_set, valid_loader=val_set, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent, save_frequency=save_frequency, save_best=save_best)        
elif(run_mode==2 and IsResume == True):
    test_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, test_path_fully, None, None, n_workers_test, getROIRes, undersampling_mat, filename_filter_test, ds_split_xlsx_test, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_test)
    model.Test(dataloader=test_set, output_path=output_path)
elif(run_mode==3):
    train_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, train_path_fully, None, None, n_workers_train, getROIRes, undersampling_mat, filename_filter_train, ds_split_xlsx_train, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_train)
    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path, save_frequency=save_frequency)

    test_set, _ = model.IntitializeMRITorchDS(dsClass, None, None, None, test_path_fully, None, None, n_workers_test, getROIRes, undersampling_mat, filename_filter_test, ds_split_xlsx_test, corupt_type, mask_func, mask_seed, top_n_sampled=top_n_sampled_test)
    model.Test(dataloader=test_set, output_path=output_path, sliceno = slicenoTest)
elif(run_mode==4):
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

#########################

from support.ResultAnalyzerExtended import AnalyzeResultFolder
AnalyzeResultFolder(output_path, '', True, True)

if(modelType_actual == '2DMultiSlice'):
    from support.MultisliceCombiner import MultisliceCombiner
    MultisliceCombiner(output_path, output_path+'-Combined', True, True)
    AnalyzeResultFolder(output_path+'-Combined', '', True, True)


