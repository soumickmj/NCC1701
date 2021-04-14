import os
import inspect
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.Helper import Helper, EvaluationParams
import utils.MRITorchDSTransforms as transformsMRI
import utils.fastMRI.TorchDSTransforms as transformsFastMRI
from utils.TorchDS.MATHandles.MAT import MRIMATTorchDS 
from utils.TorchDS.fastMRIv2 import MRITorchDS as fastMRIDS #TODO: Hotfix
from utils.TorchDS.TorchDSInitializer import TorchDSInitializer
from support.save_recons import ValidateNStoreRecons
from utils.fastMRI.fastmri_utils import ValidateNStoreRecons as fastMRIReconStore

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing. Hyperparameters needs to be dynamic"

class Engine(object):
    """It is the Main class for the Engine (Model Core)"""

    def __init__(self, domain, batchSize = 1, imageSize = None, type = '2DMultislice', pin_memory = False, IsRadial = False, 
                    undersamplingMask = None, log_path = None, log_freq = 10, use_amp=False):
        """Constructor of the Engine. Initializes certain values"""
              
        self.domain = domain
        self.dsdomain = domain
        if "_" in  self.domain:
            self.domain = self.domain.split("_")[-1]
        self.batchSize = batchSize 
        self.imageSize = imageSize 
        self.type = type 
        self.pin_memory = pin_memory 
        self.IsRadial = IsRadial
        self.helpOBJ = Helper(domain) #To use functions from the Model Helper class
        self.IsTrained = False #To be set to True when the model is trained for atleast one epoch  
        self.use_amp = use_amp

        if undersamplingMask is not None:
            self.undersamplingMask = torch.from_numpy(undersamplingMask).float()
            self.missingMask = (~(self.undersamplingMask.byte())).float()
        else:
            self.undersamplingMask = None
            self.missingMask = None

        if log_path is not None:
            self.tb_writer = SummaryWriter(log_dir = log_path)
        else:
            self.tb_writer = SummaryWriter()
        self.log_freq = log_freq

        self.gradscaler = GradScaler(enabled=use_amp)

    def CreateModel(self, netModel, device, IsCuda = False, IsMultiGPU = False, n_channels = 1, 
                    initialLearningRate = 0.001, lrScheduler_func=None, lrScheduler_param_dict=None, betas = (0.9, 0.999), epsilon = 1e-08, 
                    loss_func = nn.MSELoss, optimizer=optim.Adam, NetType = 0, IsNegetiveLoss=True, loss_type=0):
        """This function creates the model (both generator and discriminator). Initializes everything (i.e. criterion (loss function), optimizers)"""
        
        self.IsCuda = IsCuda
        self.n_channels = n_channels

        if inspect.isclass(netModel): # We will create the network object, if not already supplied
            if self.batchSize > 1:
                do_batchnorm = True
            else:
                do_batchnorm = False

            if NetType == 2:
                self.net = netModel(n_channels, forwardV=0, do_batchnorm=do_batchnorm) #TODO args forwardV
            else:
                self.net = netModel(n_channels)
        else:
            self.net = netModel 
        self.net.apply(self.InitializeWeights) # We initialize all the weights of its neural network.

        if inspect.isclass(loss_func): # We will create the loss function object, if not already supplied
            self.loss_func = loss_func()
        else:
            self.loss_func = loss_func

        self.loss_type = loss_type #0: Calculate as it is. 1: Convert to Image Space and then calculate loss 2: 0+1 dual loss (not yet implimented) 
        self.IsNegetiveLoss = IsNegetiveLoss
        self.NetType = NetType
        
        if self.IsCuda:
            self.toDevice(device, IsMultiGPU)

        self.optimizer = optimizer(self.net.parameters(), lr = initialLearningRate, betas = betas, eps = epsilon) # We create the optimizer object of the generator.
        if lrScheduler_func:
            self.lrScheduler = lrScheduler_func(self.optimizer, *lrScheduler_param_dict)
        else:
            self.lrScheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=1) #To ignore LR Decay

    def InitializeWeights(self, m):
        """Initializes Weights for our networks.
        Currently it's only for Convolutino and Batch Normalization"""

        classname = m.__class__.__name__
        if classname.find('VAE') != -1:
            return #Don't do anything if it is a VAE
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def toDevice(self, device, IsMultiGPU):
        self.device = device
        self.IsMultiGPU = IsMultiGPU
        self.net.to(device)
        self.loss_func.to(device)
        if IsMultiGPU:
            self.net = nn.DataParallel(self.net) #This makes use of all the GPUs. 