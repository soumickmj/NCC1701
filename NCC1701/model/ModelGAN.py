#!/usr/bin/env python

"""
This is the main module of our FirstGAN model. 
It creates the GAN, Trains, Tests, Validates
It also helps to initialize a new MRITorchDS for our model

Hyperparameters are currently static.

"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.Helper import Helper, EvaluationParams
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

    def __init__(self, domain, batchSize = 1, imageSize = None, type = '2DMultislice', pin_memory = False):
        """Constructor of the Model. Initializes certain values
        type: Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice
        """
              
        self.domain = domain
        self.batchSize = batchSize 
        self.imageSize = imageSize 
        self.type = type 
        self.pin_memory = pin_memory 
        self.best_accuracy = 0 #Stores the best accuracy (SSIM) during TrainNValidate
        self.helpOBJ = Helper(domain) #To use functions from the Model Helper class
        self.IsTrained = False #To be set to True when the model is trained for atleast one epoch  

    def ToCUDA(self, IsMultiGPU = True):
        """This funciton helps to convert all Tensors (including models) of this Model Class from CPU Tensors to CUDA tensors"""

        self.netG = self.netG.cuda().contiguous()
        self.netD = self.netD.cuda().contiguous()
        self.criterion = self.criterion.cuda().contiguous()
        self.IsCuda = True
        self.IsMultiGPU = IsMultiGPU

    def ToCPU(self):
        """This funciton helps to convert all Tensors (including models) of this Model Class from CUDA Tensors to CPU tensors"""

        self.netG = self.netG.cpu().contiguous()
        self.netD = self.netD.cpu().contiguous()
        self.criterion = self.criterion.cpu().contiguous()
        IsCuda = False
        IsMultiGPU = False

    def IntitializeMRITorchDS(self, dsClass, folder_path_fully, folder_path_under, extension_under, num_workers, getROIMode):
        """Function to Initialize Custom MRI DS in Torch DS format.
           Images has to be in Nifit format, folder structure is according to the OASIS1 standard"""

        # Creating the transformations
        if(self.type == '2DSingleSlice' or self.type == '2DMultiSlice'):
            transform = transforms.Compose([transformsMRI.MinMaxNormalization(),transformsMRI.ToTensor3D(),transformsMRI.ConvertToSuitableType(type=self.type),]) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        else:
            transform = transforms.Compose([transformsMRI.MinMaxNormalization(),transformsMRI.ToTensor3D(),]) # We create a list of transformations (3dto2d, tensor conversion) to apply to the input images.
        
        
        # Loading the dataset
        dataset = dsClass(folder_path_fully,folder_path_under,extension_under,domain=self.domain,transform=transform,getROIMode=getROIMode)
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

    def CreateModel(self, netModelG, netModelD, IsCuda = False, IsMultiGPU = False, n_channels = 1, 
                    initialLearningRateG = 0.001, lrDecayNEpochG = 4, lrDecayRateG = 0.5, betasG = (0.9, 0.999), epsilonG = 1e-08,
                    initialLearningRateD = 0.001, lrDecayNEpochD = 4, lrDecayRateD = 0.5, betasD = (0.9, 0.999), epsilonD = 1e-08):
        """This function creates the model (both generator and discriminator). Initializes everything (i.e. criterion (loss function), optimizers)"""

        self.IsCuda = IsCuda
        self.IsMultiGPU = IsMultiGPU
        self.n_channels = n_channels

        self.netG = netModelG(n_channels) # We create the generator object.
        self.netG.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        if(self.IsCuda):
            self.netG = self.netG.cuda()
            if(self.IsMultiGPU):
                self.netG = nn.DataParallel(self.netG) #This makes use of all the GPUs.  
                #This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension.

        # Creating the discriminator
        self.netD = netModelD(n_channels) # We create the discriminator object.
        self.netD.apply(self.InitializeWeights) # We initialize all the weights of its neural network.
        if(self.IsCuda):
            self.netD = self.netD.cuda()
            if(self.IsMultiGPU):
                self.netD = nn.DataParallel(self.netD)

        self.criterion = nn.MSELoss() # L1 Loss -  We create a criterion object that will measure the error between the prediction and the target.
        self.criterion_consistency = nn.L1Loss() # L1 Loss -  We create a criterion object that will measure the error between the prediction and the target. ##################NEW
        
        if(self.IsCuda):
            self.criterion.cuda() 
            self.criterion_consistency.cuda() ###################EW
      
        #Adam optimizer - for Generator
        #net.parameters() - Paramters of our NN
        #lr - Learning Rate
        #betas - coefficients used for computing running averages of gradient and its square
        self.optimizerG = optim.Adam(self.netG.parameters(), lr = initialLearningRateG, betas = betasG, eps = epsilonG) # We create the optimizer object of the generator.
        self.initialLearningRateG = initialLearningRateG
        self.lrDecayNEpochG = lrDecayNEpochG
        self.lrDecayRateG = lrDecayRateG

        #Adam optimizer - for Discriminator
        self.optimizerD = optim.Adam(self.netD.parameters(), lr = initialLearningRateD, betas = betasD, eps = epsilonD) # We create the optimizer object of the generator.
        self.initialLearningRateD = initialLearningRateD
        self.lrDecayNEpochD = lrDecayNEpochD
        self.lrDecayRateD = lrDecayRateD
                   
    def Train(self, dataloader = None, total_n_epoch = 25, start_epoch = 0, root_path = '', sliceno = 50):
        """This function Trains the model (generator + disciminator) for certain number of epochs, stores a checkpoint after every epoch but doesn't perform any validation
        start_epoch is usally 0. But if we are resuming training, then this can be anything.
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""

        if dataloader is None:
            dataloader = self.dataloader #load dataloader for inside of the same model if no dataloader is supplied
        for epoch in range(start_epoch, total_n_epoch): # We iterate over n_epoch epochs.
            self.TrainOneEpoch(dataloader, epoch, total_n_epoch, sliceno = sliceno) #Train for one epoch

            #Save checkpoint after training on on epoch
            checkpoint_path = os.path.join(root_path, 'checkpoints') 
            self.helpOBJ.save_checkpoint({
                'epoch': epoch + 1,
                #'model': self,
				'modelD': self.netD,
				'modelG': self.netG,
                'state_dictD': self.netD.state_dict(),
                'state_dictG': self.netG.state_dict(),
                'best_accuracy': 0, #because it is only trained, not validated
                'optimizerD' : self.optimizerD.state_dict(),
                'optimizerG' : self.optimizerG.state_dict(),
            }, False, checkpoint_path)
    
    def TrainOneEpoch(self, dataloader, epoch, n_epoch, sliceno = 50):
        """This funciton trains the model for one epoch. This is never directly called by called using either Train function or TrainNValidate function
        sliceno: is to be only used during model type 2DSingleSlice for choosing which slice no to use"""

        #Initialize Evaluation Parameters
        batch_time = EvaluationParams()
        data_time = EvaluationParams()

        # switch to train mode
        self.netD.train()
        self.netG.train()

        self.helpOBJ.adjust_learning_rate(self.optimizerG, epoch, self.initialLearningRateG, self.lrDecayNEpochG, self.lrDecayRateG)
        self.helpOBJ.adjust_learning_rate(self.optimizerD, epoch, self.initialLearningRateD, self.lrDecayNEpochD, self.lrDecayRateD)

        end = time.time()
        for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (training).
            #i: index of the loop
            #data: mini batch of Images, based on batch_size supplied to the dataloader constructor - contains images and labels
            #dataloader, 0 :collection of all Images, 0 is being the starting index  
        
            # measure data loading time
            data_time.update(time.time() - end)          
            
            # Read both Fully and Under
            fully = data['fully']
            under = data['under']

            # 1st Step: Updating the weights of the neural network of the discriminator
            #self.netD.zero_grad()
            self.optimizerD.zero_grad()
            # Training the discriminator with a real (fully-sampled) image of the dataset
            ones = torch.ones(fully.size()[0], 1)
            if(self.IsCuda):
                fully = fully.contiguous()
                fully = fully.cuda(async=True)
                ones = ones.contiguous()
                ones = ones.cuda(async=True)

            input = Variable(fully) # We wrap it in a variable. - Converting images to Torch Variabel - a special type of variable which contains both images and gradients
            target = Variable(ones) # We get the target.
            #We need to set labels to 1, as these are actual Images. So, we create a array with all ones of the size input.size()[0] (no of image in the mini batch)
            output = self.netD(input) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_fully = self.criterion(output, target) # We compute the loss between the predictions (output) and the target (equal to 1).

            # Training the discriminator with the output images generated by the generator using undersampled images
            zeros = torch.zeros(input.size()[0], 1)
            if(self.IsCuda):
                under = under.contiguous()
                under = under.cuda(async=True)
                zeros = zeros.contiguous()
                zeros = zeros.cuda(async=True)
            under = Variable(under)
            #As the first I/P of generator takes 100 random values, so we create random noise of the nize no_of_images & 100
            #input.size()[0] - mini batch size
            #100:  no of elements
            #1: & 1: This will be like 100 feature maps of size 1x1 - These 1's are for fake dimension
            generated = self.netG(under) # We forward propagate the undersampled images into the neural network of the generator to get some "improved" images.
            target = Variable(zeros) # We get the target.
            #We need to set labels to 0, as these are Fake (undersampled) Images. So, we create a array with all zeros of the size input.size()[0] (no of image in the mini batch)
            output = self.netD(generated.detach()) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            #generated.detach() - detach the gradients from the torch variable - we don't need them, because we don't care about them we only care about the stochastic gradient descent
            # detach to avoid training G on these labels
            errD_generated = self.criterion(output, target) # We compute the loss between the prediction (output) and the target (equal to 0).

            
            self.errG_consistency = self.criterion_consistency(generated, input) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1). ################NEW

            # Backpropagating the total error
            self.errD = errD_fully + errD_generated # We compute the total error of the discriminator.
            self.errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
            self.optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

            # 2nd Step: Updating the weights of the neural network of the generator
            self.optimizerG.zero_grad()
            #self.netG.zero_grad() # We initialize to 0 the gradients of the generator with respect to the weights.
            ones = torch.ones(input.size()[0], 1)
            if(self.IsCuda):
                ones = ones.contiguous()
                ones = ones.cuda(async=True)
            target = Variable(ones) # We get the target.
            #We need to set labels to 1, as these are Fake Images. And this time, the target of the fake images should be 1 as we are going to train the generator now
            output = self.netD(generated) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            self.errG_GAN = self.criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
            self.errG = self.errG_consistency + self.errG_GAN  ################NEW
            self.errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
            self.optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), self.errD.data.item(), self.errG.data.item())) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
            #print('[%d/%d][%d/%d] Loss: %.4f' % ((epoch+1), n_epoch, i, len(dataloader), self.err.data[0])) # self.err.data[0] will be depreciated, use tensor.item() to convert 0-dim tensor to python number

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
        self.netG.eval()
        self.netD.eval()

        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset (testing/prediction).
                fully = data['fully']
                under = data['under']
                under = Variable(under)
                generated = self.netG(under)
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
                #'model': self,
				'modelD': self.netD,
				'modelG': self.netG,
                'state_dictD': self.netD.state_dict(),
                'state_dictG': self.netG.state_dict(),
                'best_accuracy': self.best_accuracy,
                'optimizerD' : self.optimizerD.state_dict(),
                'optimizerG' : self.optimizerG.state_dict(),
            }, is_best, checkpoint_path)