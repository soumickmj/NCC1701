import sys
import os
import inspect
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchcomplex
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.extern.pytorch_msssim import SSIM

from Bridge.AuxiliaryEngine import Helper, EvaluationParams
from utils.TorchDS.MATHandles.MAT import MRIMATTorchDS 
from utils.TorchDS.TorchDSInitializer import TorchDSInitializer
from utils.support.save_recons import Saver
from utils.math.misc import root_sum_of_squares_pyt
from utils.utilities import tensorboard_images


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
        else:
            self.undersamplingMask = None

        self.saver = Saver(domain=domain, mask=self.undersamplingMask, isRadial=self.IsRadial)#, convertTo2D=False,getSSIMMap=True,channelSum=True,doComparisons=True,roi_crop_res=None,use_data_consistency=True) #TODO: params

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
            #TODO: uncomment stuff
            self.net = netModel(n_channels)#, forwardV=0, do_batchnorm=do_batchnorm) #TODO args forwardV
        else:
            self.net = netModel 
        self.net.apply(self.InitializeWeights) # We initialize all the weights of its neural network.

        if inspect.isclass(loss_func): # We will create the loss function object, if not already supplied
            self.loss_func = loss_func()
        else:
            self.loss_func = loss_func

        self.loss_type = loss_type #0: Calculate as it is. 1: Convert to Image Space and then calculate loss 2: 0+1 dual loss (not yet implimented) 
        self.IsNegetiveLoss = IsNegetiveLoss
        self.best_accuracy = -np.inf
        self.NetType = NetType
        
        if self.IsCuda:
            self.toDevice(device, IsMultiGPU)

        self.optimizer = optimizer(self.net.parameters(), lr = initialLearningRate, betas = betas, eps = epsilon) # We create the optimizer object of the generator.
        if lrScheduler_func:
            #TODO 
            self.lrScheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1, last_epoch=-1)
            # self.lrScheduler = lrScheduler_func(self.optimizer, *lrScheduler_param_dict)
        else:
            self.lrScheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=1) #To ignore LR Decay

        self.accuracy = SSIM(data_range=1, size_average=False, channel=n_channels, nonnegative_ssim=True, spatial_dims=2) #TODO: have other accuracy options and make spatial_dims params

    def InitializeWeights(self, m):
        """Initializes Weights for our networks.
        Currently it's only for Convolutino and Batch Normalization"""

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if type(m.weight) is nn.ParameterList or m.weight.dtype is torch.cfloat: 
                torchcomplex.nn.init.trabelsi_standard_(m.weight, kind="glorot")
            else:
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
            self.net = nn.DataParallel(self.net) #This makes use of all the GPUs. TODO: Use distributed data parallel 

    def IntitializeMRITorchDS(self, dsClass, sliceno=None, startingSelectSlice=None, endingSelectSlice=None, folder_path_fully=None, 
                              folder_path_under=None, extension_under=None, num_workers=None, getROIMode=None, undersampling_mask=None, 
                              filename_filter=None, ds_split_xlsx=None, corupt_type=None, mask_func=None, mask_seed=None, top_n_sampled=None, 
                              transform=None, ds_split2use=None):
        if dsClass is MRIMATTorchDS: #TODO: for H5 dataset as well
            dataset = MRIMATTorchDS(folder_path_fully, transform) #TODO make shuffle param
            dataloader = DataLoader(dataset, batch_size = self.batchSize, shuffle = True, num_workers = num_workers, pin_memory = self.pin_memory)
        else:
            dsInitializer = TorchDSInitializer()
            dataloader, dataset = dsInitializer.InitializeDS(domain=self.dsdomain, isRadial=self.IsRadial, sliceHandleType=self.type, 
                                                             batchSize=self.batchSize, pin_memory=self.pin_memory, dsClass=dsClass, 
                                                             sliceno=sliceno, startingSelectSlice=startingSelectSlice, endingSelectSlice=endingSelectSlice, 
                                                             folder_path_fully=folder_path_fully, folder_path_under=folder_path_under, 
                                                             extension_under=extension_under, num_workers=num_workers, getROIMode=getROIMode, 
                                                             undersampling_mask=undersampling_mask, filename_filter=filename_filter, splitXSLX=ds_split_xlsx,
                                                             split2use=ds_split2use, corupt_type=corupt_type, mask_func=mask_func, mask_seed=mask_seed,
                                                             top_n_sampled=top_n_sampled, transform=transform)
        return dataloader, dataset

    def criterion(self, out, gt):
        if self.loss_type == 0:
            return self.loss_func(out, gt)
        elif self.loss_type == 1 or self.loss_type == 2:
            g = self.helpOBJ.Domain2Image_pyt(out)
            f = self.helpOBJ.Domain2Image_pyt(gt)
            if self.loss_type == 1:
                return self.loss_func(torch.abs(g), torch.abs(f))
            elif self.loss_type == 2:
                return self.loss_func(torch.abs(g), torch.abs(f)) + self.loss_func(torch.angle(g), torch.angle(f))
            else:
                sys.exit("MainEngine: Invalid loss_type")

    def Train(self, dataloader = None, total_n_epoch = 25, start_epoch = 0, root_path = '', save_frequency = 10):
        """This function Trains the model (generator + disciminator) for certain number of epochs, stores a checkpoint after every epoch but doesn't perform any validation
        start_epoch is usally 0. But if we are resuming training, then this can be anything."""

        if dataloader is None:
            dataloader = self.dataloader #load dataloader for inside of the same model if no dataloader is supplied TODO
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.             
            avg_loss = self.TrainOneEpoch(dataloader, epoch, total_n_epoch) #Train for one epoch

            self.tb_writer.add_scalar('Train/AvgLossEpoch', avg_loss, epoch)

            if(epoch+1 == total_n_epoch): #if this is the last epoch, doesn't matter what is the freuqency, have to save the model
                save_frequency = 1

            #Save checkpoint after training on on epoch
            checkpoint_path = os.path.join(root_path, 'checkpoints') 
            if self.IsMultiGPU:
                net = self.net.module
            else:
                net = self.net
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'model': net,
                'state_dict': net.state_dict(),
                'best_accuracy': 0, #because it is only trained, not validated
                'optimizer' : self.optimizer.state_dict(),
                'AMPScaler' : self.gradscaler.state_dict(),
                'LRScheduler' : self.lrScheduler.state_dict(),
            }, False, checkpoint_path, save_frequency)

    def TrainOneEpoch(self, dataloader, epoch, n_epoch):
        """This funciton trains the model for one epoch. This is never directly called by called using either Train function or TrainNValidate function"""

        #Initialize Evaluation Parameters
        batch_time = EvaluationParams()
        data_time = EvaluationParams()
        losses = EvaluationParams()

        # switch to train mode
        self.net.train()

        runningLosses = EvaluationParams()
        end = time.time()
        for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (training)
            data_time.update(time.time() - end) # measures the data loading time
            gt = Variable(data['gt']).to(self.device)
            inp = Variable(data['inp']).to(self.device)

            if(self.type == '2DMultiSlice' or self.type == '2DSelectMultiSlice'):
                #TODO
                sys.exit("2DMultiSlice or 2DSelectMultiSlice in TrainOneEpoch not implimented")

            self.optimizer.zero_grad() #Set the parameter gradients to zero
            
            with autocast(enabled=self.use_amp):
                if "custom_forward" in dir(self.net):
                    self.err, _ = self.net.custom_forward(inp=inp, gt=gt, loss_func=self.criterion, neg_loss=self.IsNegetiveLoss)
                else:
                    out = self.net(inp) 
                    if self.IsNegetiveLoss:
                        self.err = -self.criterion(out, gt)
                    else:
                        self.err = self.criterion(out, gt)

            self.gradscaler.scale(self.err).backward() #TODO: can handle only one loss. Add custombackawrd to model for multiple loss
            self.gradscaler.step(self.optimizer)
            self.lrScheduler.step()
            self.gradscaler.update()
            
            if self.IsNegetiveLoss:
                losses.update(round(-self.err.data.item(),4))
                runningLosses.update(round(-self.err.data.item(),4))
            else:
                losses.update(round(self.err.data.item(),4))
                runningLosses.update(round(self.err.data.item(),4))

            print('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), runningLosses.val[-1]))
            if i % self.log_freq == 0:
                niter = epoch*len(dataloader)+i
                self.tb_writer.add_scalar('Train/Loss', runningLosses.stat(), niter) 
                runningLosses.reset()        

            batch_time.update(time.time() - end) # measures the batch time
            end = time.time() #resets the timer

        torch.cuda.empty_cache()
        self.IsTrained = True
        return round(losses.stat(), 4) 

    def Test(self, dataloader, output_path):
        """This function performs testing on a trained network. Tt is similar to validate, just deference being it calls Validate function and sets saveOutput to True
        """
        
        self.Validate(dataloader, saveOutput=True, print_freq=self.log_freq, epoch=None, n_epoch=None, output_path=output_path, IsTest=True)

    def Validate(self, dataloader, saveOutput=False, print_freq=10, epoch=None, n_epoch=None, output_path=None, IsTest=False):
        """This function validates the performance of a trained model. 
        This is never directly called, but called from either Test (on test set - saving the output) or TrainNValidate (after training on each epoch - doesn't save the output, just returns the average accuracy for this epoch
        """

        if IsTest:
            tb_tag_type = 'Test/'
        else:
            tb_tag_type = 'Validate/'       

        data_time = EvaluationParams()
        batch_time = EvaluationParams()
        losses = EvaluationParams() 
        accuracies = EvaluationParams()

        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(dataloader, 0):
                data_time.update(time.time() - end) # measures the data loading time
                gt = Variable(data['gt']).to(self.device)
                inp = Variable(data['inp']).to(self.device)

                if(self.type == '2DMultiSlice' or self.type == '2DSelectMultiSlice'):
                    #TODO
                    sys.exit("2DMultiSlice or 2DSelectMultiSlice in Validate not implimented")

                with autocast(enabled=self.use_amp):
                    if "custom_forward" in dir(self.net):
                        self.err, out = self.net.custom_forward(inp=inp, gt=gt, loss_func=self.criterion, neg_loss=self.IsNegetiveLoss)
                    else:
                        out = self.net(inp) 
                        if self.IsNegetiveLoss:
                            self.err = -self.criterion(out, gt)
                        else:
                            self.err = self.criterion(out, gt)   

                if self.use_amp:
                    out = out.type(gt.dtype)

                if self.IsNegetiveLoss:
                    losses.update(round(-self.err.data.item(),4))
                else:
                    losses.update(round(self.err.data.item(),4))

                if self.domain == 'fourier' or self.domain == 'hartley':
                    # inp_ = self.helpOBJ.Domain2Image_pyt(inp, return_abs=True)
                    out_ = self.helpOBJ.Domain2Image_pyt(out, return_abs=True)
                    gt_ = self.helpOBJ.Domain2Image_pyt(gt, return_abs=True)
                else:
                    # inp_ = inp
                    out_ = out
                    gt_ = gt
                
                accuracies.update(list(self.accuracy(out_, gt_).cpu().numpy()))

                if IsTest:
                    print('[%d/%d] Test Loss: %.4f' % (i, len(dataloader), losses.val[-1]))
                else:
                    print('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), losses.val[-1]))

                if epoch is not None:
                    niter = epoch*len(dataloader)+i
                else:
                    niter = i
                self.tb_writer.add_scalar(tb_tag_type+'Loss', losses.stat(), niter) 
                self.tb_writer.add_scalar(tb_tag_type+'Accuracy', accuracies.stat(), niter)
                if i % self.log_freq == 0: #Saving all logs, but saving only nth image, determined by self.log_freq
                    tensorboard_images(self.tb_writer, None, root_sum_of_squares_pyt(out_.detach(), dim=1, keepdim=True), None, niter, tb_tag_type[:-1])

                if saveOutput:  #This usually to be used with Test function 
                    for j in range(0, len(data['subjectName'])): 
                        path4output = os.path.join(output_path, data['subjectName'][j], data['fileName'][j]) 
                        mask = data['mask'][j] if "mask" in data else None
                        RefK = data['gt_ksp'][j] if "gt_ksp" in data else None
                        UndersampledK = data['inp_ksp'][j] if "inp_ksp" in data else None
                        self.saver.ValidateNStore(out.data[j],gt[j],inp[j],mask,path4output,
                                                    OutK=None,RefK=RefK,UndersampledK=UndersampledK) #OutK is to be passed if we create some net which gives output of kSP as well as ImgSpace
                batch_time.update(time.time() - end) # measures the batch time
                end = time.time() #resets the timer

        torch.cuda.empty_cache()
        return accuracies.stat(), losses.stat()

    def TrainNValidate(self, dataset = None, train_loader=None, valid_loader=None, output_path = 'results', total_n_epoch = 25, start_epoch = 0, num_workers = 0, valid_percent = 0.25, save_frequency=10, save_best=True):
        """This function creates a training and validation set out of the training set and then first trains then validates for each epoch
        start_epoch is usally 0. But if we are resuming training, then this can be anything. 
        Size of the validation set is defined by valid_percent param"""

        #Create train and validation set from the given training dataset
        if train_loader is None or valid_loader is None:
            train_loader, valid_loader = self.helpOBJ.GetTrainValidateLoader(dataset=dataset, batchSize=self.batchSize, num_workers=num_workers, valid_percent=valid_percent)
        
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.
            #Train for one epoch
            avg_loss_train = self.TrainOneEpoch(train_loader, epoch, total_n_epoch) #Train for one epoch

            self.tb_writer.add_scalar('Train/AvgLossEpoch', avg_loss_train, epoch)

            #Evaluate on validation set
            avg_accuracy_val, avg_loss_val = self.Validate(dataloader=valid_loader, saveOutput=False, print_freq=10, epoch=epoch, n_epoch=total_n_epoch, IsTest=False)

            self.tb_writer.add_scalar('Validate/AvgLossEpoch', avg_loss_val, epoch)
            self.tb_writer.add_scalar('Validate/AvgAccuracyEpoch', avg_accuracy_val, epoch)            

            if epoch+1 == total_n_epoch: #if this is the last epoch, doesn't matter what is the freuqency, have to save the model
                save_frequency = 1

            # remember best accuracy and save checkpoint
            if save_best: 
                is_best = avg_accuracy_val > self.best_accuracy
            else:
                is_best = False
            self.best_accuracy = max(avg_accuracy_val, self.best_accuracy)
            checkpoint_path = os.path.join(output_path, 'checkpoints') 
            if self.IsMultiGPU:
                net = self.net.module
            else:
                net = self.net
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'model': net,
                'state_dict': net.state_dict(),
                'best_accuracy': self.best_accuracy,
                'optimizer' : self.optimizer.state_dict(),
                'AMPScaler' : self.gradscaler.state_dict(),
                'LRScheduler' : self.lrScheduler.state_dict(),
            }, is_best, checkpoint_path, save_frequency)