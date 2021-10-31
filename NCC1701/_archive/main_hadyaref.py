#!/usr/bin/env python

"""
This is the main module, entry point of our program.
Settings of parameters are implimented here.
Hyperparameter updates can/should be done here

For reference regarding remote debugging, visit: https://docs.microsoft.com/en-us/visualstudio/python/debugging-python-code-on-remote-linux-machines
for VSCode support, https://code.visualstudio.com/docs/python/python-tutorial and https://donjayamanne.github.io/pythonVSCodeDocs/docs/debugging_remote-debugging/
"""


import os
import sys
import tkinter as tk
from tkinter import filedialog
import socket
import importlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pytorch_ssim
from utils.HandleTorchModel import GetModelMemoryUsage
from model.Helper import Helper

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
##For updating parameters
gpuID = "0"
maskORom_path = r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat"
slicenoTest = 50 #Only for 2DSingleSlice
startingSelectSlice = 51 #Only for 2DSelectMultiSlice and 3DSelectSlice
endingSelectSlice = 80 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DSingleSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
modelType4Actual = '2DSingleSlice'
CheckpointPath = r"E:\TrainedModels\Enterprise_Gen2\Resnet2Dv2b14\OASIS_Brains\Final_Checkpoint\n40_slice51to80_Varden1D30.pth.tar"
tb_logroot = r"E:\Datasets\ResNetTestSet\Skyra\TBLogs"
train_path_fully = '/home/schatter/Dataset/MATs/OASIS-Subset1-Varden1D30SameMask-Slice51/ds1-fourier-train'
train_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Train'
train_entension_under = 'nii.gz'
test_path_fully = r'E:\Datasets\OASIS-Lesion\MATs\1DVarden30Mask'
test_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Test'
test_entension_under = 'nii.gz'
output_path = r"E:\Datasets\OASIS-Lesion\Output_MICAIResnet"
output_folder = 'Varden1D30'
IsRadial = False
####Updating parameters zone ends here



neumann_name = "c502.vc-a"

import scipy.io as sio
mat_contents = sio.loadmat(maskORom_path)
mask = mat_contents['mask']

#### Basic parameters
domain = 'image' #In which domain, network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image |
if(domain == 'fourier' or domain == 'real_fourier' or domain == 'fourier_magphase'):
    n_channels = 2
else:
    n_channels = 1
#n_channels = 6 #For Brain Mask, Will only work in image domain for now. TODO: Brain mask for frquency domains as well

slicenoTrain = 50 #Only for 2DSingleSlice
slicenoTrain = slicenoTrain - 1 #As array starts from 0
slicenoTest = slicenoTest - 1 #As array starts from 0
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
IsGAN = False

run_mode = 2 #0: Train 1: Train and Validate 2:Test 3: Train followed by Test, 4: Train and Validate followed by Test
IsResume = True

#### CUDA Related Parameters
IsCuda = False
cudnn.benchmark = IsCuda
IsMultiGPU = False


#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.MATHandles.MAT').MRIMATTorchDS #For DS following OASIS1 Standard

#### Net Model Import path or object of the net model itself. Incase of GANs, this will be the Generator
netModel = importlib.import_module('model.ResNet.Resnet2Dv2b14').ResNet 
##OR can directly pass the object of the class - This can be used if non-default parameters (except for n_channels) are to be sent to the consturctor
#from model.UNet.Unet2Dv1 import UNet
#netModel = UNet(n_channels=n_channels, depth=5,start_filts=32, up_mode='transpose', merge_mode='concat')

#### Load the custom model class
if IsGAN:
    Model = importlib.import_module('model.ModelGAN').Model  
else:   
    Model = importlib.import_module('model.ModelSSIMLoss').Model 

##### Parameters related to training, testing and validation
batch_size = 1
n_workers_train = 0
n_workers_test = 0
n_epoch = 50
valid_percent=0.20 #Percentage of the complete dataset to be used for creating the validation set.
getROIMode=None #It can be 'None' if we don't need to crop out non-ROI, or 'FromUnderToFully' or 'FromFullyToUnder'
pinMemory = False
loss_func = pytorch_ssim.SSIM

#### Parameters related to the optimizer. Incase of GAN, these will be used for the Generator
initialLearningRate = 0.0001#0.0002 #0.0001 #0.1
lrDecayNEpoch = 50 #4 #Decay the learning rate after every 4th epoch
lrDecayRate = 0.1 #Decay divide by 10 #0.5 #Decay by half
beta1 = 0.9 #0.5 #0.9 #Momentum parameter of the gradients
beta2 = 0.999 #Momentum parameter of the gradients squared 
epsilon = 1e-09 #Small float added to avoid division by zero

#### Parameters related to the optimizer of the Discriminator. Only used incase of GAN
initialLearningRateD = 0.0001#0.1
lrDecayNEpochD = 50#4 #Decay the learning rate after every 4th epoch
lrDecayRateD = 0.1#0.5 #Decay by half
beta1D = 0.9 #0.9 #Momentum parameter of the gradients
beta2D = 0.999 #Momentum parameter of the gradients squared 
epsilonD = 1e-09 #Small float added to avoid division by zero

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=gpuID


#### Define path to dataset and also to output
output_path = os.path.join(output_path, output_folder)
tb_logpath = os.path.join(tb_logroot, output_folder)

######################


######################
##Main heart of the code

model = Model(domain, batch_size, type = modelType, pin_memory = pinMemory, undersamplingMask = mask, log_path=tb_logpath)
if IsGAN:
    model.CreateModel(netModelG=netModel, netModelD=netModelD, IsCuda=IsCuda, IsMultiGPU=IsMultiGPU, n_channels=n_channels, 
                      initialLearningRateG=initialLearningRate, lrDecayNEpochG=lrDecayNEpoch, lrDecayRateG=lrDecayRate, betasG=(beta1, beta2), epsilonG=epsilon,
                      initialLearningRateD=initialLearningRateD, lrDecayNEpochD=lrDecayNEpochD, lrDecayRateD=lrDecayRateD, betasD=(beta1D, beta2D), epsilonD=epsilonD)
else:
    model.CreateModel(netModel, IsCuda = IsCuda, IsMultiGPU = IsMultiGPU, n_channels=n_channels, 
                      initialLearningRate=initialLearningRate, lrDecayNEpoch=lrDecayNEpoch, lrDecayRate=lrDecayRate, betas=(beta1, beta2), epsilon=epsilon, loss_func=loss_func)
if(IsResume): #If IsResume is set to true, then resume from a given path or ask for a checkpoint path
    if(CheckpointPath is ''):
        root = tk.Tk()
        root.withdraw()
        CheckpointPath = filedialog.askopenfilename()
    if IsGAN:
        model, start_epoch = model.helpOBJ.load_checkpointGAN(checkpoint_file = CheckpointPath, model=model)
    else:
        model, start_epoch = model.helpOBJ.load_checkpoint(checkpoint_file = CheckpointPath, model=model, IsCuda=IsCuda)
    if model.IsCuda != IsCuda: #If IsCuda status changed from last time
        if(IsCuda):
            model.ToCUDA(IsMultiGPU)
        else:
            model.ToCPU()
else:   
    start_epoch = 0

#Run according to the mentioned run mode
if(run_mode == 0):
    train_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask)
    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path)
elif(run_mode==1):
    _, train_setDS = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask) 
    model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent)
elif(run_mode==2 and IsResume == True):
    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
    model.Test(dataloader=test_set, output_path=output_path)
elif(run_mode==3):
    train_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask)
    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path)

    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
    model.Test(dataloader=test_set, output_path=output_path, sliceno = slicenoTest)
elif(run_mode==4):
    _, train_setDS = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask) 
    model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent)

    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
    model.Test(dataloader=test_set, output_path=output_path)
else:
    print('Invalid Run Mode. Remember: Test mode requires IsResume set to true')

#########################

from support.ApplyDataConsistancy import ApplyDataConsis
ApplyDataConsis(output_path, IsRadial, True, maskORom_path)

from support.ResultAnalyzerExtended import AnalyzeResultFolder
AnalyzeResultFolder(output_path, '', True, True)

if(modelType4Actual == '2DMultiSlice'):
    from support.MultisliceCombiner import MultisliceCombiner
    MultisliceCombiner(output_path, output_path+'-Combined', True, True)
    AnalyzeResultFolder(output_path+'-Combined', '', True, True)