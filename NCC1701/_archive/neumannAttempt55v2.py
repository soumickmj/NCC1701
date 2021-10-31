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
import socket
import importlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pytorch_ssim
from utils.HandleTorchModel import GetModelMemoryUsage
from model.Helper import Helper
from utils.GetFreeGPU import get_free_gpu

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

neumann_name = "c502.vc-a"

import scipy.io as sio
mat_contents = sio.loadmat(r"/scratch/tmp/schatter/Code/EnterpriseV1/Gen1/v3-20181020/1DVardenMask.mat")
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
slicenoTest = 50 #Only for 2DSingleSlice
slicenoTest = slicenoTest - 1 #As array starts from 0
startingSelectSlice = 51 #Only for 2DSelectMultiSlice and 3DSelectSlice
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
endingSelectSlice = 80 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DSingleSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
IsGAN = False

run_mode = 4 #0: Train 1: Train and Validate 2:Test 3: Train followed by Test, 4: Train and Validate followed by Test
IsResume = False
CheckpointPath = r"/scratch/tmp/schatter/Output/Neural/EnterpriseV1-gen2/Attempt48-WithValidate-Radial30GA-Resnet2Dv2SSIMLoss-Slice50/checkpoints/checkpoint.pth.tar"

#### CUDA Related Parameters
IsCuda = True
cudnn.benchmark = IsCuda
IsMultiGPU = True


#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.MATHandles.MAT').MRIMATTorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS

#### Net Model Import path or object of the net model itself. Incase of GANs, this will be the Generator
#netModel = importlib.import_module('model.ConvTrans.Translator2D').Translator 
#netModel = importlib.import_module('model.ConvTrans.Translator2Dv2').Translator  
#netModel = importlib.import_module('model.ConvTrans.Translator3D').Translator  
#netModel = importlib.import_module('model.ConvTrans.Translator3DFastDebug').Translator  
#netModel = importlib.import_module('model.ConvTrans.Translator3DIncludesTranspose').Translator  
#netModel = importlib.import_module('model.UNet.Unet2Dv1').UNet
#netModel = importlib.import_module('model.UNet.Unet2Dv2').UNet 
#netModel = importlib.import_module('model.UNet.Unet2Dv23D').UNet 
#netModel = importlib.import_module('model.UNet.Unet2DvMario1').UNet 
#netModel = importlib.import_module('model.UNet.Unet3Dv1').UNet 
#netModel = importlib.import_module('model.UNet.Unet3Dv2').UNet 
#netModel = importlib.import_module('model.UNet.DenseUNet2Dv1').DenseUNet 
#netModel = importlib.import_module('model.GAN.Resnet2DSigF').ResNet 
netModel = importlib.import_module('model.ResNet.Resnet2Dv2b14').ResNet 
#netModel = importlib.import_module('model.DenseNet.DenseResNet').DenseResNet 
#netModel = importlib.import_module('model.DenseNet.Densenet2D').DenseNet 
#netModel = importlib.import_module('model.DenseNet.DanseNet2DReference').DenseNet3 
##OR can directly pass the object of the class - This can be used if non-default parameters (except for n_channels) are to be sent to the consturctor
#from model.UNet.Unet2Dv1 import UNet
#netModel = UNet(n_channels=n_channels, depth=5,start_filts=32, up_mode='transpose', merge_mode='concat')

#### Net Model Import path or object of the net model itself - for the Generator. Only to be used during GANs
netModelD = importlib.import_module('model.GAN.Discriminator2D').Discriminator 
#netModelD = importlib.import_module('model.GAN.Discriminator3D').Discriminator
#netModelD = importlib.import_module('model.GAN.Discriminator3DFastDebug').Discriminator 
##OR can directly pass the object of the class - This can be used if non-default parameters (except for n_channels) are to be sent to the consturctor
#from model.GAN.Discriminator2D import Discriminator
#netModelD = Discriminator(n_channels=n_channels)

#### Load the custom model class
if IsGAN:
    Model = importlib.import_module('model.ModelGAN').Model  
else:   
    Model = importlib.import_module('model.ModelSSIMLoss').Model 

##### Parameters related to training, testing and validation
batch_size = 15
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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


#### Define path to dataset and also to output - based on machine name
pc_name = socket.gethostname()
if(pc_name == 'BMMR-Soumick'):
    ##For BMMR-Soumick - For Undersampled Images. Fot MAT DSs, only the path given for fully is utilized
    tb_logroot = r'D:\CloudData\OneDrive\OvGU\Tensorboard_Logs'
    train_path_fully = r'D:\Datasets\MATs\20190320Fresh\OASIS-Subset1-Varden30SameMask-Slice50\ds1-image-train'
    train_path_under = r'D:\Datasets\OASIS1\Undersampled-Registered\TrainSet\varden30SameMask'
    train_entension_under = 'nii.gz'
    test_path_fully = r'D:\Datasets\MATs\20190320Fresh\OASIS-Subset1-Varden30SameMask-Slice50\ds1-image-test'
    test_path_under = r'D:\Datasets\OASIS1\Undersampled-Registered\TestSet\varden30SameMask'
    test_entension_under = 'nii.gz'
    output_path = r'D:\Output'
    output_folder = 'Attempt30_3-WithValidate-OASISVarden30SameMask_Fresh-Resnet2Dv2NoDropSSIMLoss-SingleSlice50'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)

    ##For BMMR-Soumick - For Contrast Translation
    #train_path_fully = r'D:\Datasets\AlexMultiCon2\IXI-Dataset\T2\Guys-Train'
    #train_path_under = r'D:\Datasets\AlexMultiCon2\IXI-Dataset\T1\Guys-Train'
    #train_entension_under = 'nii.gz'
    #test_path_fully = r'D:\Datasets\AlexMultiCon2\IXI-Dataset\T2\Guys-Test'
    #test_path_under = r'D:\Datasets\AlexMultiCon2\IXI-Dataset\T1\Guys-Test'
    #test_entension_under = 'nii.gz'
    #output_path = 'D:\Output\AlexT1T2'
    #output_folder = 'scratch'
    #output_path = os.path.join(output_path, output_folder)
elif(pc_name == 'powerzon04'):
    ##For Mario's PC - For Contrast Translation
    #train_path_fully = r'D:\Soumick\AlexMultiCon2\IXI-Dataset\T2\Guys-Train'
    #train_path_under = r'D:\Soumick\AlexMultiCon2\IXI-Dataset\T1\Guys-Train'
    #train_entension_under = 'nii.gz'
    #test_path_fully = r'D:\Soumick\AlexMultiCon2\IXI-Dataset\T2\Guys-Test'
    #test_path_under = r'D:\Soumick\AlexMultiCon2\IXI-Dataset\T1\Guys-Test'
    #test_entension_under = 'nii.gz'
    #output_path = 'D:\Soumick\Output\AlexT1T2-New'
    #output_folder = 'Attempt14-DenseUnet2Dv1'
    #output_path = os.path.join(output_path, output_folder)

	##For Mario's PC - For Radial
    #train_path_fully = r'D:\Datasets\OASIS-Sub\Train\full'
    #train_path_under = r'D:\Datasets\OASIS-Sub\Train\spokes60'
    #train_entension_under = 'nii'
    #test_path_fully = r'D:\Datasets\OASIS-Sub\Test\full'
    #test_path_under = r'D:\Datasets\OASIS-Sub\Test\spokes60'
    #test_entension_under = 'nii'
    #output_path = 'D:\Soumick\Output\OASIS1spokes60'
    #output_folder = 'Attempt3-Resnet34-2DSingleSlice'
    #output_path = os.path.join(output_path, output_folder)

    ##For Mario's PC - undersampled Alex3 varden20
    train_path_fully = r'D:\Soumick\AlexMultiCon3\Repro-DeepLearning\axial\T2\Vol2SORTED\Train'
    train_path_under = r'D:\Datasets\AlexMultiCon3\under\axial\T2\Vol2\varden20SORTED\Train'
    train_entension_under = 'nii.gz'
    test_path_fully = r'D:\Soumick\AlexMultiCon3\Repro-DeepLearning\axial\T2\Vol2SORTED\Test'
    test_path_under = r'D:\Datasets\AlexMultiCon3\under\axial\T2\Vol2\varden20SORTED\Test'
    test_entension_under = 'nii.gz'
    output_path = 'D:\Soumick\Output\AlexDS3T2-varden20'
    output_folder = 'Attempt1-Resnet34-2DSingleSlice'
    output_path = os.path.join(output_path, output_folder)
elif(pc_name == 'quadro'):
    ##For Qadro - For Contrast Translation
    tb_logroot = '/home/soumick/Output/Neural/TensorBoardLogs'
    train_path_fully = '/home/soumick/Datasets/Datasets/MATs/20190320Fresh_Correct/OASIS-AllCombined4-51to80SingleEach/ds1-image-train'
    train_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Train'
    train_entension_under = 'nii.gz'
    test_path_fully = '/home/soumick/Datasets/Datasets/MATs/20190320Fresh_Correct/OASIS-AllCombined4-51to80SingleEach/ds1-image-test'
    test_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Test'
    test_entension_under = 'nii.gz'
    output_path = '/home/soumick/Output/Neural/EnterpriseV1-gen2/'
    output_folder = 'Attempt23_Redo-WithValidate-AllCombined4-Resnet2Dv2SSIMLoss-51to80SingleEach'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
elif(pc_name == neumann_name):
    ##For Neumann - For Undersampled Images
    tb_logroot = '/scratch/tmp/schatter/Output/Neural/TensorBoardLogs'
    train_path_fully = '/scratch/tmp/schatter/Datasets/MATs/20190320Fresh/OASIS-Subset1-Varden1D30SameMask-SingleEach/ds1-image-train'
    train_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Train'
    train_entension_under = 'nii.gz'
    test_path_fully = '/scratch/tmp/schatter/Datasets/MATs/20190320Fresh/OASIS-Subset1-Varden1D30SameMask-SingleEach/ds1-image-test'
    test_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Test'
    test_entension_under = 'nii.gz'
    output_path = '/scratch/tmp/schatter/Output/Neural/EnterpriseV1-gen2/'
    output_folder = 'Attempt55_v2-WithValidate-Varden1D30SameMask-Resnet2Dv2SSIMLoss-SingleEach'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
elif(pc_name == 'gpu18.urz.uni-magdeburg.de')
    ##For GPU18 Cluster - For Undersampled Images
    tb_logroot = '/scratch/tmp/schatter/Output/Neural/TensorBoardLogs'
    train_path_fully = '/scratch/tmp/schatter/Datasets/MATs/20190320Fresh/OASIS-Subset1-Varden1D30SameMask-SingleEach/ds1-image-train'
    train_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Train'
    train_entension_under = 'nii.gz'
    test_path_fully = '/scratch/tmp/schatter/Datasets/MATs/20190320Fresh/OASIS-Subset1-Varden1D30SameMask-SingleEach/ds1-image-test'
    test_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Test'
    test_entension_under = 'nii.gz'
    output_path = '/scratch/tmp/schatter/Output/Neural/EnterpriseV1-gen2/'
    output_folder = 'Attempt55_v2-WithValidate-Varden1D30SameMask-Resnet2Dv2SSIMLoss-SingleEach'
    output_path = os.path.join(output_path, output_folder)
    tb_logpath = os.path.join(tb_logroot, output_folder)
else:
    sys.exit('PC Name Recognized. So, path definations not initialized')

######################


######################
##Main heart of the code

#model = Model(domain, batch_size, type = modelType, pin_memory = pinMemory, undersamplingMask = mask, log_path=tb_logpath)
#if IsGAN:
#    model.CreateModel(netModelG=netModel, netModelD=netModelD, IsCuda=IsCuda, IsMultiGPU=IsMultiGPU, n_channels=n_channels, 
#                      initialLearningRateG=initialLearningRate, lrDecayNEpochG=lrDecayNEpoch, lrDecayRateG=lrDecayRate, betasG=(beta1, beta2), epsilonG=epsilon,
#                      initialLearningRateD=initialLearningRateD, lrDecayNEpochD=lrDecayNEpochD, lrDecayRateD=lrDecayRateD, betasD=(beta1D, beta2D), epsilonD=epsilonD)
#else:
#    model.CreateModel(netModel, IsCuda = IsCuda, IsMultiGPU = IsMultiGPU, n_channels=n_channels, 
#                      initialLearningRate=initialLearningRate, lrDecayNEpoch=lrDecayNEpoch, lrDecayRate=lrDecayRate, betas=(beta1, beta2), epsilon=epsilon, loss_func=loss_func)
#if(IsResume): #If IsResume is set to true, then resume from a given path or ask for a checkpoint path
#    if(CheckpointPath is ''):
#        root = tk.Tk()
#        root.withdraw()
#        CheckpointPath = filedialog.askopenfilename()
#    if IsGAN:
#        model, start_epoch = model.helpOBJ.load_checkpointGAN(checkpoint_file = CheckpointPath, model=model)
#    else:
#        model, start_epoch = model.helpOBJ.load_checkpoint(checkpoint_file = CheckpointPath, model=model)
#    if model.IsCuda != IsCuda: #If IsCuda status changed from last time
#        if(IsCuda):
#            model.ToCUDA(IsMultiGPU)
#        else:
#            model.ToCPU()
#else:   
#    start_epoch = 0

##Run according to the mentioned run mode
#if(run_mode == 0):
#    train_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask)
#    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path)
#elif(run_mode==1):
#    _, train_setDS = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask) 
#    model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent)
#elif(run_mode==2 and IsResume == True):
#    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
#    model.Test(dataloader=test_set, output_path=output_path)
#elif(run_mode==3):
#    train_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask)
#    model.Train(dataloader=train_set, total_n_epoch=n_epoch, start_epoch=start_epoch, root_path=output_path)

#    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
#    model.Test(dataloader=test_set, output_path=output_path, sliceno = slicenoTest)
#elif(run_mode==4):
#    _, train_setDS = model.IntitializeMRITorchDS(dsClass, slicenoTrain, startingSelectSlice, endingSelectSlice, train_path_fully, train_path_under, train_entension_under, n_workers_train, getROIMode, mask) 
#    model.TrainNValidate(dataset=train_setDS, output_path=output_path, total_n_epoch=n_epoch, start_epoch=start_epoch, num_workers=n_workers_train,valid_percent=valid_percent)

#    test_set, _ = model.IntitializeMRITorchDS(dsClass, slicenoTest, startingSelectSlice, endingSelectSlice, test_path_fully, test_path_under, test_entension_under, n_workers_test, getROIMode, mask)
#    model.Test(dataloader=test_set, output_path=output_path)
#else:
#    print('Invalid Run Mode. Remember: Test mode requires IsResume set to true')

##########################

#from support.ApplyDataConsistancy import ApplyDataConsis

#ApplyDataConsis(output_path, False, True, r'/scratch/tmp/schatter/Code/EnterpriseV1/Gen2/v1-20190322/1DVarden30Mask.mat')

#from support.MultisliceCombiner import MultisliceCombiner

#MultisliceCombiner(output_path, output_path+'-Combined', True, True)

from support.ResultAnalyzerExtended import AnalyzeResultFolder

AnalyzeResultFolder(output_path, '', True, True)
AnalyzeResultFolder(output_path+'-Combined', '', True, True)


