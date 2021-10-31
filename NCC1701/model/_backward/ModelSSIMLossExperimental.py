#!/usr/bin/env python

"""
This is the main module of our ConvolutionalTranslator model. 
It creates the Convolutional Translator, Trains, Tests, Validates
It also helps to initialize a new MRITorchDS for our model

Hyperparameters are currently static.

"""

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
from tensorboardX import SummaryWriter
from model.Helper import Helper, EvaluationParams
import utils.MRITorchDSTransforms as transformsMRI
from utils.TorchDS.MATHandles.MAT import MRIMATTorchDS 
from utils.TorchDS.TorchDSInitializer import TorchDSInitializer
from support.save_recons import ValidateNStoreRecons

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing. Hyperparameters needs to be dynamic"


class Model(object):
    """It is the Main class for the Model (Model Core) - Inlcudes both Generator and Discriminator"""

    def __init__(self, domain, batchSize = 1, imageSize = None, type = '2DMultislice', pin_memory = False, IsRadial = False, undersamplingMask = None, log_path = None):
        """Constructor of the Model. Initializes certain values
        type: Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice
        """
              
        self.domain = domain
        self.batchSize = batchSize 
        self.imageSize = imageSize 
        self.type = type 
        self.pin_memory = pin_memory 
        self.IsRadial = IsRadial
        self.best_accuracy = 0 #Stores the best accuracy (SSIM) during TrainNValidate
        self.helpOBJ = Helper(domain) #To use functions from the Model Helper class
        self.IsTrained = False #To be set to True when the model is trained for atleast one epoch  

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
        self.log_freq = 10

    def ToCUDA(self, IsMultiGPU = True):
        """This funciton helps to convert all Tensors (including models) of this Model Class from CPU Tensors to CUDA tensors"""

        self.net = self.net.cuda().contiguous()
        self.criterion = self.criterion.cuda().contiguous()
        self.IsCuda = True
        self.IsMultiGPU = IsMultiGPU

    def ToCPU(self):
        """This funciton helps to convert all Tensors (including models) of this Model Class from CUDA Tensors to CPU tensors"""

        self.net = self.net.cpu().contiguous()
        self.criterion = self.criterion.cpu().contiguous()
        IsCuda = False
        IsMultiGPU = False

    #def IntitializeMRITorchDS(self, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode, undersampling_mask=None):
    #    """Function to Initialize Custom MRI DS in Torch DS format.
    #       Images has to be in Nifit format, folder structure is according to the OASIS1 standard"""
        
    #    if(self.domain == 'image'):
    #        listOfTransforms = [transformsMRI.MinMaxNormalization(), transformsMRI.ToTensor3D()]
    #    else:
    #        listOfTransforms = [transformsMRI.ToTensor3D(),]

    #    if(self.IsRadial):
    #        listOfTransforms.insert(0, transformsMRI.RemoveVerySmallEValues(delta=0.1))

    #    if(self.type == '2DSingleSlice'):
    #        listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=self.type,sliceno=sliceno)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
    #    elif(self.type == '2DSelectMultiSlice' or self.type == '3DSelectSlice'):
    #        listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=self.type, startingSelectSlice=startingSelectSlice, endingSelectSlice=endingSelectSlice)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
    #    elif(self.type == '2DMultiSlice'):
    #        listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=self.type)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        
    #    transform = transforms.Compose(listOfTransforms)
        
    #    # Loading the dataset
    #    dataset = dsClass(folder_path_fully,folder_path_under,extension_under,domain=self.domain,transform=transform,getROIMode=getROIMode,undersampling_mask=undersampling_mask)
    #    dataloader = DataLoader(dataset, batch_size = self.batchSize, shuffle = True, num_workers = num_workers, pin_memory = self.pin_memory)
        
    #    return dataloader, dataset

    def IntitializeMRITorchDS(self, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode, undersampling_mask=None):
        """Function to Initialize Custom MRI DS in Torch DS format.
           Images has to be in Nifit format, folder structure is according to the OASIS1 standard"""
        
        if dsClass is MRIMATTorchDS:
            #AS for MATTorchDS only one path needed, so the path for fully is been utilized
            dataloader, dataset = self.IntitializeMRIMATTorchDS(folder_path_fully, num_workers)
        else:
            dsInitializer = TorchDSInitializer()
            dataloader, dataset = dsInitializer.InitializeDS(self.domain, self.IsRadial, self.type, self.batchSize, self.pin_memory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode, undersampling_mask)
        return dataloader, dataset

    def IntitializeMRIMATTorchDS(self, folder_path, num_workers):
        """Function to Initialize Custom MRI DS in Torch DS format, from MAT Files already been created from any Custom MRI Torch DS.
           This reads MAT files, present in single folder
           This is a pure function, which can be directly called or called from IntitializeMRITorchDS function"""
        
        # Loading the dataset
        dataset = MRIMATTorchDS(folder_path)
        dataloader = DataLoader(dataset, batch_size = self.batchSize, shuffle = True, num_workers = num_workers, pin_memory = self.pin_memory)

        return dataloader, dataset
    
    def InitializeWeights(self, m):
        """Initializes Weights for our networks.
        Currently it's only for Convolutino and Batch Normalization"""

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def CreateModel(self, netModel, IsCuda = False, IsMultiGPU = False, n_channels = 1, initialLearningRate = 0.001, lrDecayNEpoch = 4, lrDecayRate = 0.5, betas = (0.9, 0.999), epsilon = 1e-08, loss_func = nn.MSELoss, optimizer=optim.Adam, IsImgKSPNet = False):
        """This function creates the model (both generator and discriminator). Initializes everything (i.e. criterion (loss function), optimizers)"""

        self.IsCuda = IsCuda
        self.IsMultiGPU = IsMultiGPU
        self.n_channels = n_channels

        if self.IsCuda and self.undersamplingMask is not None:
            self.undersamplingMask = self.undersamplingMask.cuda()
            self.missingMask = self.missingMask.cuda()

        self.IsImgKSPNet = IsImgKSPNet
        if IsImgKSPNet:
            if inspect.isclass(netModel):# We create the generator object
                self.net = netModel(n_channels, self.undersamplingMask)
            else:
                self.net = netModel 
                self.net.underMask = self.undersamplingMask
                self.net.missingMask = self.missingMask
        else:
            if inspect.isclass(netModel):# We create the generator object
                self.net = netModel(n_channels)
            else:
                self.net = netModel 
        self.net.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        self.criterion = loss_func() # MSE Loss -  We create a criterion object that will measure the error between the prediction and the target.
        
        if self.IsCuda and not self.IsImgKSPNet:
            self.net.cuda()
            self.criterion.cuda()
            if(self.IsMultiGPU):
                self.net = nn.DataParallel(self.net) #This makes use of all the GPUs.  
      
        #Adam optimizer
        #net.parameters() - Paramters of our NN
        #lr - Learning Rate
        #betas - coefficients used for computing running averages of gradient and its square
        self.optimizer = optimizer(self.net.parameters(), lr = initialLearningRate, betas = betas, eps = epsilon) # We create the optimizer object of the generator.
        self.initialLearningRate = initialLearningRate
        self.lrDecayNEpoch = lrDecayNEpoch
        self.lrDecayRate = lrDecayRate
                   
    def Train(self, dataloader = None, total_n_epoch = 25, start_epoch = 0, root_path = '', save_frequency = 10):
        """This function Trains the model (generator + disciminator) for certain number of epochs, stores a checkpoint after every epoch but doesn't perform any validation
        start_epoch is usally 0. But if we are resuming training, then this can be anything."""

        if dataloader is None:
            dataloader = self.dataloader #load dataloader for inside of the same model if no dataloader is supplied
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.             
            avg_loss = self.TrainOneEpoch(dataloader, epoch, total_n_epoch) #Train for one epoch

            self.tb_writer.add_scalar('Train/AvgLossEpoch', avg_loss, epoch)

            if(epoch+1 == total_n_epoch): #if this is the last epoch, doesn't matter what is the freuqency, have to save the model
                save_frequency = 1

            #Save checkpoint after training on on epoch
            checkpoint_path = os.path.join(root_path, 'checkpoints') 
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'model': self.net,
                'state_dict': self.net.state_dict(),
                'best_accuracy': 0, #because it is only trained, not validated
                'optimizer' : self.optimizer.state_dict(),
            }, False, checkpoint_path, save_frequency)
    
    def TrainOneEpoch(self, dataloader, epoch, n_epoch):
        """This funciton trains the model for one epoch. This is never directly called by called using either Train function or TrainNValidate function"""

        #Initialize Evaluation Parameters
        batch_time = EvaluationParams()
        data_time = EvaluationParams()
        losses = EvaluationParams()

        # switch to train mode
        self.net.train()

        self.helpOBJ.adjust_learning_rate(self.optimizer, epoch, self.initialLearningRate, self.lrDecayNEpoch, self.lrDecayRate)

        end = time.time()
        runningLoss = 0
        runningLossCounter = 0
        for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (training).
            #i: index of the loop
            #data: mini batch of Images, based on batch_size supplied to the dataloader constructor - contains images and labels
            #dataloader, 0 :collection of all Images, 0 is being the starting index            

            #Set the parameter gradients to zero
            self.optimizer.zero_grad()
        
            # measure data loading time
            data_time.update(time.time() - end)     
            fully = data['fully']
            under = data['under']

            if(self.type == '2DMultiSlice' or self.type == '2DSelectMultiSlice'):
                fully = fully.permute(1,0,2,3,4)
                fully = fully.view(fully.size(0),fully.size(1)*fully.size(2),fully.size(3),fully.size(4)).permute(1,0,2,3)
                under = under.permute(1,0,2,3,4)
                under = under.view(under.size(0),under.size(1)*under.size(2),under.size(3),under.size(4)).permute(1,0,2,3)

            fully = Variable(fully)
            under = Variable(under)
            if(self.IsCuda):
                fully = fully.contiguous()
                under = under.contiguous()
                #fully = fully.cuda(async=True)
                #under = under.cuda(async=True) #async no more in use. Use non_blocking
                fully = fully.cuda()
                under = under.cuda()
            
            if self.IsImgKSPNet:
                generated, out_img_net, out_ksp_net_k, fully_k = self.net(fully) # We forward propagate the undersampled images into the neural network of the generator to get some "improved" images.
                final_ssim = self.criterion(generated, fully)
                img_net_ssim = self.criterion(out_img_net, fully)
                ksp_net_loss = nn.MSELoss()(out_ksp_net_k, fully_k)
                self.err = ksp_net_loss - img_net_ssim - final_ssim
            else:
                generated = self.net(under) # We forward propagate the undersampled images into the neural network of the generator to get some "improved" images.
                #generated = self.helpOBJ.applyDataConsistancy(under, generated, self.missingMask) #TODO: Create option for controlling it
                self.err = -self.criterion(generated, fully) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
            self.err.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
            self.optimizer.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #Updated losses
            losses.update(round(-self.err.data.item(),4))
            runningLoss += round(-self.err.data.item(),4)
            runningLossCounter += 1

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

            #For tensorboard
            if i % self.log_freq == 0:
                niter = epoch*len(dataloader)+i
                self.tb_writer.add_scalar('Train/Loss', runningLoss/runningLossCounter, niter)
                runningLoss = 0
                runningLossCounter = 0

            print('[%d/%d][%d/%d] Loss: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), -self.err.data.item())) # We print les losses of the convolutional translator.
            #print('[%d/%d][%d/%d] Loss: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), self.err.data[0])) # self.err.data[0] will be depreciated, use tensor.item() to convert 0-dim tensor to python number

        self.IsTrained = True
        return round(losses.avg, 4)
        
    def Test(self, dataloader, output_path):
        """This function performs testing on a trained network. Tt is similar to validate, just deference being it calls Validate function and sets saveOutput to True
        """
        
        self.Validate(dataloader, saveOutput=True, print_freq=self.log_freq, epoch=None, n_epoch=None, output_path=output_path, IsTest=True)
        
    def Validate(self, dataloader, saveOutput=False, print_freq=10, epoch=None, n_epoch=None, output_path=None, IsTest=False):
        """This function validates the performance of a trained model. 
        This is never directly called, but called from either Test (on test set - saving the output) or TrainNValidate (after training on each epoch - doesn't save the output, just returns the average accuracy for this epoch
        """

        if(IsTest):
            tb_tag_type = 'Test/'
        else:
            tb_tag_type = 'Validate/'

        #Initialize Params
        if(self.type == '2DSingleSlice'):
            transformTensorToNumpy = transformsMRI.FromTensorToNumpy2D()
        else:
            transformTensorToNumpy = transformsMRI.FromTensorToNumpy3D()        
        batch_time = EvaluationParams()
        losses = EvaluationParams() #Currently not in use
        accuracies = EvaluationParams()

        # switch to evaluate mode
        self.net.eval()

        with torch.no_grad():
            end = time.time()

            runningLoss = 0
            runningAcc = 0
            runningLossCounter = 0
            for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (testing/prediction).

                fully = data['fully']
                under = data['under']
                if(self.type == '2DMultiSlice' or self.type == '2DSelectMultiSlice'):
                    under = under.permute(1,0,2,3,4)
                    under = under.view(under.size(0),under.size(1)*under.size(2),under.size(3),under.size(4)).permute(1,0,2,3)
                under = Variable(under)
                if(self.IsCuda):
                    under = under.contiguous()
                    under = under.cuda()
                    fully = fully.contiguous()
                    fully = fully.cuda()
                if self.IsImgKSPNet:
                    generated, out_img_net, out_ksp_net_k, fully_k = self.net(fully) # We forward propagate the undersampled images into the neural network of the generator to get some "improved" images.
                    final_ssim = self.criterion(generated, fully)
                    img_net_ssim = self.criterion(out_img_net, fully)
                    ksp_net_loss = nn.MSELoss()(out_ksp_net_k, fully_k)
                    loss = ksp_net_loss - img_net_ssim - final_ssim
                else:
                    generated = self.net(under)
                    loss = -self.criterion(generated, fully)

                if(self.type == '2DMultiSlice' or self.type == '2DSelectMultiSlice'):
                    generated = generated.permute(1,0,2,3)
                    generated = generated.view(generated.size(0),self.batchSize,generated.size(1)/self.batchSize,generated.size(2),generated.size(3)).permute(1,0,2,3,4) #If n_channels is more than 1
                    under = under.permute(1,0,2,3)
                    under = under.view(under.size(0),self.batchSize,under.size(1)/self.batchSize,under.size(2),under.size(3)).permute(1,0,2,3,4) #If n_channels is more than 1
                
                
                losses.update(-round(loss.data.item(),4))

                #If the tensor were CUDA tensors, convert them to CPU tensors
                if(self.IsCuda):
                    under = under.cpu()
                    generated = generated.cpu()
                    fully = fully.cpu()

                #generated_corrected = self.helpOBJ.applyDataConsistancy(under, generated, self.missingMask)                
                #generated = self.helpOBJ.applyDataConsistancy(under, generated, self.missingMask) #TODO: Create option for controlling it. Also, create option for saving both generated and corrected

                # Performance measurement                
                #prec1, prec5 = self.helpOBJ.accuracy(generated, fully, topk=(1, 5))
                acc = self.helpOBJ.accuracy(generated, fully, transformTensorToNumpy)                
                accuracies.update(acc, len(acc))

                #To help in tensorboard logging process, find average
                acc = sum(acc) / len(acc) 
                runningLoss += -round(loss.data.item(),4)
                runningAcc += acc
                runningLossCounter += 1

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          '\nTime Sum: {batch_time.sum:.3f} Avg: ({batch_time.avg:.3f})'
                          '\nLoss {loss.val} ({loss.avg:.4f})'
                          '\nAccuracy (SSIM) {accuracies.val} ({accuracies.avg:.3f})\n\n'.format(
                           i, len(dataloader), batch_time=batch_time, loss=losses,
                           accuracies=accuracies))

                    #For tensorboard
                    if(epoch is not None):
                        niter = epoch*len(dataloader)+i
                    else:
                        niter = i
                    self.tb_writer.add_scalar(tb_tag_type+'Loss', runningLoss/runningLossCounter, niter)
                    self.tb_writer.add_scalar(tb_tag_type+'Accuracy', runningAcc/runningLossCounter, niter)
                    runningLoss = 0
                    runningAcc = 0
                    runningLossCounter = 0

                #This usually to be used with Test function
                if(saveOutput):    
                    #Validate and Save the generated images
                    for j in range(0, len(data['subjectName'])): 
                        path4output = os.path.join(output_path, data['subjectName'][j], data['fileName'][j]) 
                        generatedImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(generated.data[j]))
                        fullyImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(fully[j]))
                        underImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(under[j]))
                        ValidateNStoreRecons(generatedImg,fullyImg,underImg,path4output)
            
            if(epoch is not None):
                print('Evaluation over for epoch ['+str(epoch+1)+'/'+str(n_epoch)+']')
            else:
                print('Evaluation Done\n')
            print('Average SSIM {accuracies.avg:.3f}'.format(accuracies=accuracies))

        return accuracies.avg, round(losses.avg, 4)
    
    def TrainNValidate(self, dataset = None, output_path = 'results', total_n_epoch = 25, start_epoch = 0, num_workers = 0, valid_percent = 0.25, save_frequency=10, save_best=True):
        """This function creates a training and validation set out of the training set and then first trains then validates for each epoch
        start_epoch is usally 0. But if we are resuming training, then this can be anything. 
        Size of the validation set is defined by valid_percent param"""

        #Create train and validation set from the given training dataset
        train_loader, valid_loader = self.helpOBJ.GetTrainValidateLoader(dataset=dataset, batchSize=self.batchSize, num_workers=num_workers, valid_percent=valid_percent)
        
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.

            #Train for one epoch
            avg_loss_train = self.TrainOneEpoch(train_loader, epoch, total_n_epoch) #Train for one epoch

            self.tb_writer.add_scalar('Train/AvgLossEpoch', avg_loss_train, epoch)

            #Evaluate on validation set
            avg_accuracy_val, avg_loss_val = self.Validate(dataloader=valid_loader, saveOutput=False, print_freq=10, epoch=epoch, n_epoch=total_n_epoch, IsTest=False)

            self.tb_writer.add_scalar('Validate/AvgLossEpoch', avg_loss_train, epoch)
            self.tb_writer.add_scalar('Validate/AvgAccuracyEpoch', avg_accuracy_val, epoch)            

            if(epoch+1 == total_n_epoch): #if this is the last epoch, doesn't matter what is the freuqency, have to save the model
                save_frequency = 1

            # remember best accuracy and save checkpoint
            if save_best:
                is_best = avg_accuracy_val > self.best_accuracy
            else:
                is_best = False
            self.best_accuracy = max(avg_accuracy_val, self.best_accuracy)
            checkpoint_path = os.path.join(output_path, 'checkpoints') 
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'model': self.net,
                'state_dict': self.net.state_dict(),
                'best_accuracy': self.best_accuracy,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best, checkpoint_path)
