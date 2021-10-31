#!/usr/bin/env python

"""
This is the main module of our ConvolutionalTranslator model. 
It creates the Convolutional Translator, Trains, Tests, Validates
It also helps to initialize a new MRITorchDS for our model

Hyperparameters are currently static.

This program translates from one domain to another

In Case of Undersampled images,
Domain A: Under
Domain B: Fully
In case of Contrast Translation,
Domain A: Source Contrast
Domain B: Destination Contrast

"""

import os
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.Helper import Helper, EvaluationParams
from model.GAN.CycleGANUtils import *
import utils.MRITorchDSTransforms as transformsMRI
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

    def __init__(self, domain, batchSize = 1, imageSize = None, type = '2DMultislice', pin_memory = False, IsRadial = False):
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

    def IntitializeMRITorchDS(self, dsClass, sliceno, folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode):
        """Function to Initialize Custom MRI DS in Torch DS format.
           Images has to be in Nifit format, folder structure is according to the OASIS1 standard"""
        
        listOfTransforms = [transformsMRI.MinMaxNormalization(), transformsMRI.ToTensor3D()]

        if(self.IsRadial):
            listOfTransforms.insert(0, transformsMRI.RemoveVerySmallEValues(delta=0.1))

        if(self.type == '2DSingleSlice'):
            listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=self.type,sliceno=sliceno)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        elif(self.type == '2DMultiSlice'):
            listOfTransforms.append(transformsMRI.ConvertToSuitableType(type=self.type)) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        
        transform = transforms.Compose(listOfTransforms)
        
        # Loading the dataset
        dataset = dsClass(folder_path_fully,folder_path_under,extension_under,domain=self.domain,transform=transform,getROIMode=getROIMode)
        dataloader = DataLoader(dataset, batch_size = self.batchSize, shuffle = True, num_workers = num_workers, pin_memory = self.pin_memory)
        
        return dataloader, dataset
    
    def InitializeWeights(self, m):
        """Initializes Weights for our networks.
        Currently it's only for Convolutino and Batch Normalization"""

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def CreateModel(self, netModelG, netModelD, IsCuda = False, IsMultiGPU = False, n_resnet_block = 9, img_height = 256, img_width = 256, n_channels = 1, n_epochs = 200, start_epoch = 0, decay_start_epoch = 100, initialLearningRate = 0.0002, betas = (0.5, 0.999)):
        """This function creates the model (both generator and discriminator). Initializes everything (i.e. criterion (loss function), optimizers)"""

        self.IsCuda = IsCuda
        self.IsMultiGPU = IsMultiGPU
        self.n_channels = n_channels

        # Initialize generator and discriminator
        self.G_AB = netModelG(n_channels=n_channels, res_blocks=n_resnet_block) ######################TODO
        self.G_BA = netModelG(n_channels=n_channels, res_blocks=n_resnet_block)#####################TODO: uncomment it dynamically
        self.D_A = netModelD(n_channels=n_channels)
        self.D_B = netModelD(n_channels=n_channels)
        self.G_AB.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        self.G_BA.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        self.D_A.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        self.D_B.apply(self.InitializeWeights) # We initialize all the weights of its neural network.

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        if(self.IsCuda):
            self.G_AB = self.G_AB.cuda()
            self.G_BA = self.G_BA.cuda()
            self.D_A = self.D_A.cuda()
            self.D_B = self.D_B.cuda()
            self.criterion_GAN.cuda()
            self.criterion_cycle.cuda()
            self.criterion_identity.cuda()
            #if(self.IsMultiGPU): TODO:: Multi GPU
            #    self.net = nn.DataParallel(self.net) #This makes use of all the GPUs. 
        
        # Loss weights
        self.lambda_cyc = 10
        self.lambda_id = 0.5 * self.lambda_cyc

        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=initialLearningRate, betas=betas)
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=initialLearningRate, betas=betas)
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=initialLearningRate, betas=betas)
      
        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_start_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_start_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_start_epoch).step)

        # Create a Tensor of float type, based on Cuda settings
        self.Tensor = torch.cuda.FloatTensor if IsCuda else torch.Tensor

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Calculate output of image discriminator (PatchGAN)
        self.patchD = (1, img_height // 2**4, img_width // 2**4)
                   
    def Train(self, dataloader = None, total_n_epoch = 25, start_epoch = 0, root_path = '', sliceno = 50):
        """This function Trains the model (generator + disciminator) for certain number of epochs, stores a checkpoint after every epoch but doesn't perform any validation
        start_epoch is usally 0. But if we are resuming training, then this can be anything.
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""

        if dataloader is None:
            dataloader = self.dataloader #load dataloader for inside of the same model if no dataloader is supplied

        self.prev_time = time.time()
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.
            self.TrainOneEpoch(dataloader, epoch, total_n_epoch, sliceno = sliceno) #Train for one epoch

            #Save checkpoint after training on on epoch
            checkpoint_path = os.path.join(root_path, 'checkpoints') 
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'G_AB': self.G_AB.state_dict(),
                'G_BA': self.G_BA.state_dict(),
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'best_accuracy': 0, #because it is only trained, not validated
                #'optimizer_G' : self.optimizer_G.state_dict(),
                #'optimizer_D_A' : self.optimizer_D_A.state_dict(),
                #'optimizer_D_B' : self.optimizer_D_B.state_dict(),
            }, False, checkpoint_path)
    
    def TrainOneEpoch(self, dataloader, epoch, n_epoch, sliceno = 50):
        """This funciton trains the model for one epoch. This is never directly called by called using either Train function or TrainNValidate function
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""

        for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (training).
            #i: index of the loop
            #data: mini batch of Images, based on batch_size supplied to the dataloader constructor - contains images and labels
            #dataloader, 0 :collection of all Images, 0 is being the starting index   
            
            # Set model input
            real_A = data['fully'].type(self.Tensor)
            real_B = data['under'].type(self.Tensor)

            valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patchD))), requires_grad=False)
            fake = Variable(self.Tensor(np.ones((real_A.size(0), *self.patchD))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            #Set the parameter gradients to zero
            self.optimizer_G.zero_grad()

             # Identity loss
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity

            loss_G.backward()
            self.optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            #Set the parameter gradients to zero
            self.optimizer_D_A.zero_grad()

            # Real loss
            loss_real = self.criterion_GAN(self.D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            self.optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            #Set the parameter gradients to zero
            self.optimizer_D_B.zero_grad()

            # Real loss
            loss_real = self.criterion_GAN(self.D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            self.optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epoch * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - self.prev_time))
            self.prev_time = time.time()

            # Print log
            sys.stdout.write("\r\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, n_epoch,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

        self.IsTrained = True
        
    def Test(self, dataloader, output_path, sliceno=50):
        """This function performs testing on a trained network. Tt is similar to validate, just deference being it calls Validate function and sets saveOutput to True
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""
        
        self.Validate(dataloader, saveOutput=True, print_freq=10, epoch=None, n_epoch=None, output_path=output_path, sliceno=sliceno)
        
    def Validate(self, dataloader, saveOutput=False, print_freq=10, epoch=None, n_epoch=None, output_path=None, sliceno=50):
        """This function validates the performance of a trained model. 
        This is never directly called, but called from either Test (on test set - saving the output) or TrainNValidate (after training on each epoch - doesn't save the output, just returns the average accuracy for this epoch
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""

        #Initialize Params
        if(self.type == '2DSingleSlice'):
            transformTensorToNumpy = transformsMRI.FromTensorToNumpy2D()
        else:
            transformTensorToNumpy = transformsMRI.FromTensorToNumpy3D()        
        batch_time = EvaluationParams()
        losses = EvaluationParams() #Currently not in use
        accuracies = EvaluationParams()

        # switch to evaluate mode
        #self.net.eval()

        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (testing/prediction).
                fully = data['fully'].type(self.Tensor)
                under = data['under'].type(self.Tensor)
                under = Variable(under)
                generated = self.G_AB(under)
                if(self.type == '2DMultiSlice'):
                    generated = generated.permute(1,0,2,3)
                    generated = generated.view(generated.size(0),self.batchSize,generated.size(1)/self.batchSize,generated.size(2),generated.size(3)).permute(1,0,2,3,4) #If n_channels is more than 1
                    under = under.permute(1,0,2,3)
                    under = under.view(under.size(0),self.batchSize,under.size(1)/self.batchSize,under.size(2),under.size(3)).permute(1,0,2,3,4) #If n_channels is more than 1
                
                #If the tensor were CUDA tensors, convert them to CPU tensors
                if(self.IsCuda):
                    fully = fully.cpu()
                    under = under.cpu()
                    generated = generated.cpu()

                # Performance measurement
                #loss = self.criterionOut(generated, fully)
                #losses.update(loss.data[0], input.size(0))
                #prec1, prec5 = self.helpOBJ.accuracy(generated, fully, topk=(1, 5))
                acc = self.helpOBJ.accuracy(generated, fully, transformTensorToNumpy)                
                accuracies.update(acc, len(acc))

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

        return accuracies.avg
    
    def TrainNValidate(self, dataset = None, output_path = 'results', total_n_epoch = 25, start_epoch = 0, num_workers = 0, valid_percent = 0.25):
        """This function creates a training and validation set out of the training set and then first trains then validates for each epoch
        start_epoch is usally 0. But if we are resuming training, then this can be anything. 
        Size of the validation set is defined by valid_percent param"""

        #Create train and validation set from the given training dataset
        train_loader, valid_loader = self.helpOBJ.GetTrainValidateLoader(dataset=dataset, batchSize=self.batchSize, num_workers=num_workers, valid_percent=valid_percent)
        
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.

            #Train for one epoch
            self.TrainOneEpoch(train_loader, epoch, total_n_epoch)

            #Evaluate on validation set
            accuracy = self.Validate(dataloader=valid_loader, saveOutput=False, print_freq=10, epoch=epoch, n_epoch=total_n_epoch)

            # remember best accuracy and save checkpoint
            is_best = accuracy > self.best_accuracy
            self.best_accuracy = max(accuracy, self.best_accuracy)
            checkpoint_path = os.path.join(output_path, 'checkpoints') 
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                'G_AB': self.G_AB.state_dict(),
                'G_BA': self.G_BA.state_dict(),
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'best_accuracy': self.best_accuracy,
                #'optimizer_G' : self.optimizer_G.state_dict(),
                #'optimizer_D_A' : self.optimizer_D_A.state_dict(),
                #'optimizer_D_B' : self.optimizer_D_B.state_dict(),
            }, is_best, checkpoint_path)
