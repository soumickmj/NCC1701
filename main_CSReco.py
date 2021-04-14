#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import socket
import importlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pytorch_ssim
from utils.HandleTorchModel import GetModelMemoryUsage
from model.Helper import Helper
from utils.GetFreeGPU import get_free_gpu


# In[ ]:


##For updating parameters
gpuID = "1"
maskORom_path = r"/mnt/MEMoRIAL/MEMoRIAL_M1.p-4/Chatterjee/Results/ResNetPaper/CodeForEval/enterprisencc1701_hadya/1DVarden10Mask.mat"
slicenoTest = 50 #Only for 2DSingleSlice #NOT NEEDED FOR WORKING WITH MATS
startingSelectSlice = 51 #Only for 2DSelectMultiSlice and 3DSelectSlice #NOT NEEDED FOR WORKING WITH MATS
endingSelectSlice = 80 #Only for 2DSelectMultiSlice and 3DSelectSlice #NOT NEEDED FOR WORKING WITH MATS
modelType4Testing = '2DMultiSlice' #2DSingleSlice important for multislice combiner
# CheckpointPath = r"S:\MEMoRIAL_SharedStorage_M1.4\Hadya\4Hadya_BMMRSoumick\Resnet2Dv2b14\IXI-T1\HH\Final_Checkpoint\h7_AllSlices_Varden1D30.pth.tar"
# CheckpointPath = r"D:\Hadya\Research Project\Juypeter\TrainedModels\Enterprise_Gen2\Resnet2Dv2b14\OASIS_Brains\Final_Checkpoint\n37_slice50_Varden1D30.pth.tar"
tb_logroot = r"/run/media/soumick/Data/ROugh/TensorBoardLogs"
test_path_fully = r"/run/media/soumick/MicksDen/Datasets/IXI/MATs/IXI-T1/Guys/Slice60to90/Varden1D10SameMask/test"
test_path_under = r"D:\Research Project\Juypeter\Dataset\Original\Undersampled\Uniform28-4step\Test" #NOT NEEDED FOR WORKING WITH MATS
test_entension_under = 'nii.gz' #NOT NEEDED FOR WORKING WITH MATS
output_path = r"/run/media/soumick/MicksDen/Results/ResNetPaper/CS/L1/IXI"
output_folder = r"Varden1D10_Guys_Slice60to90" 
IsRadial = True
RunOnly4Combine = False
CSMode = 1 #0: Sense, 1:L1 wavelet, 2: TV
####Updating parameters zone ends here


# In[ ]:

modelType = '2DSingleSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice #NOT NEEDED FOR WORKING WITH MATS

neumann_name = "c502.vc-a"

if not IsRadial: #if cartesian, then read the mask
    import scipy.io as sio
    mat_contents = sio.loadmat(maskORom_path)
    mask = mat_contents['mask']
else: #if radial, then this never been used. so just put some zeros as placeholder
    import numpy as np
    mask = np.zeros((256,256))

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
IsCuda = True
cudnn.benchmark = IsCuda
IsMultiGPU = False


#### PyTorch Custom DS Class to use - comment out the first and uncomment the second for MATs
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.MATHandles.MAT').MRIMATTorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.ADNI').MRITorchDS #For DS following ADNI

#### Net Model Import path or object of the net model itself. Incase of GANs, this will be the Generator
# netModel = importlib.import_module('model.ResNet.Resnet2Dv2b14').ResNet ##### FOR OLDER CHECKPOINTS - OASIS
netModel = importlib.import_module('model.ResNet.Resnet2Dv2b14_ADNI').ResNet  ##### FOR NEWER CHECKPOINTS - IXI/ADNI
##OR can directly pass the object of the class - This can be used if non-default parameters (except for n_channels) are to be sent to the consturctor
#from model.UNet.Unet2Dv1 import UNet
#netModel = UNet(n_channels=n_channels, depth=5,start_filts=32, up_mode='transpose', merge_mode='concat')

#### Load the custom model class
if IsGAN:
    Model = importlib.import_module('model.ModelGAN').Model  
else:   
    Model = importlib.import_module('model.ModelSSIMLoss_CS').Model 

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

import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0' #gpuID


#### Define path to dataset and also to output

train_path_fully = '/home/schatter/Dataset/MATs/OASIS-Subset1-Varden1D30SameMask-Slice51/ds1-fourier-train'
train_path_under = '/home/soumick/Datasets/AlexMultiCon2/IXI-Dataset/T1/Guys-Train'
train_entension_under = 'nii.gz'
output_path = os.path.join(output_path, output_folder)
tb_logpath = os.path.join(tb_logroot, output_folder)

######################


######################
##Main heart of the code

if not RunOnly4Combine:
    model = Model(domain, batch_size, type = modelType, pin_memory = pinMemory, undersamplingMask = mask, log_path=tb_logpath, maskORom_path=maskORom_path, CSMode=CSMode)
    if IsGAN:
        model.CreateModel(netModelG=netModel, netModelD=netModelD, IsCuda=IsCuda, IsMultiGPU=IsMultiGPU, n_channels=n_channels, 
                        initialLearningRateG=initialLearningRate, lrDecayNEpochG=lrDecayNEpoch, lrDecayRateG=lrDecayRate, betasG=(beta1, beta2), epsilonG=epsilon,
                        initialLearningRateD=initialLearningRateD, lrDecayNEpochD=lrDecayNEpochD, lrDecayRateD=lrDecayRateD, betasD=(beta1D, beta2D), epsilonD=epsilonD)
    else:
        model.CreateModel(netModel, IsCuda = IsCuda, IsMultiGPU = IsMultiGPU, n_channels=n_channels, 
                        initialLearningRate=initialLearningRate, lrDecayNEpoch=lrDecayNEpoch, lrDecayRate=lrDecayRate, betas=(beta1, beta2), epsilon=epsilon, loss_func=loss_func)
    # if(IsResume): #If IsResume is set to true, then resume from a given path or ask for a checkpoint path
    #     if(CheckpointPath is ''):
    #         root = tk.Tk()
    #         root.withdraw()
    #         CheckpointPath = filedialog.askopenfilename()
    #     if IsGAN:
    #         model, start_epoch = model.helpOBJ.load_checkpointGAN(checkpoint_file = CheckpointPath, model=model)
    #     else:
    #         model, start_epoch = model.helpOBJ.load_checkpoint(checkpoint_file = CheckpointPath, model=model, IsCuda=IsCuda)
    #     if model.IsCuda != IsCuda: #If IsCuda status changed from last time
    #         if(IsCuda):
    #             model.ToCUDA(IsMultiGPU)
    #         else:
    
    #             model.ToCPU()
    # else:   
    #     start_epoch = 0

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

    # from support.ApplyDataConsistancy import ApplyDataConsis
    # ApplyDataConsis(output_path, IsRadial, True, maskORom_path)

    from support.ResultAnalyzerExtended import AnalyzeResultFolder
    AnalyzeResultFolder(output_path, '', False, True)

if(modelType4Testing == '2DMultiSlice'):
    from support.MultisliceCombiner import MultisliceCombiner
    MultisliceCombiner(output_path, output_path+'-Combined', True, False)
    AnalyzeResultFolder(output_path+'-Combined', '', False, True)


# In[ ]:





