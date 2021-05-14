#!/usr/bin/env python

"""
This module helps to validate and store after reconsturcting.

This code is created with the help of pytorch_modelsize by jacobkimmel 
link: https://github.com/jacobkimmel/pytorch_modelsize

"""

import sys
import os
import pickle
from math import sqrt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio 
from utils.math.freq_trans import ifft2c, fft2c
from utils.math.misc import root_sum_of_squares_np
from utils.support.data_consistency import DataConsistency
from utils.HandleNifti import FileSave, Nifti3Dto2D, Nifti2Dto3D

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Nearly Finished"

class Saver():

    def __init__(self, domain="img", mask=None, isRadial=False, convertTo2D=False,getSSIMMap=True,channelSum=True,doComparisons=True,roi_crop_res=None,use_data_consistency=True,save_coil_imgs=True,save_raw=False):
        self.domain = domain
        self.isRadial = isRadial
        self.convertTo2D = convertTo2D
        self.getSSIMMap = getSSIMMap
        self.channelSum = channelSum
        self.doComparisons = doComparisons
        self.roi_crop_res = roi_crop_res
        self.use_data_consistency = use_data_consistency
        self.save_coil_imgs = save_coil_imgs
        self.save_raw = save_raw
        if self.use_data_consistency:
            self.datacon = DataConsistency(isRadial = self.isRadial, mask=mask)

    def ValidateNStore(self,Out,Ref,Undersampled,mask,path4output,OutK=None,RefK=None,UndersampledK=None):
        """
        This is a common function, called by the two following functions (ValidateNStoreRecons & ValidateNStoreReconsROI), not to be called directly
        This function validates the output againest a reference image (fully sampled), also checks how much the image have improved from the input image (undersampled)
        Parameters:
        Out: Output of the algorithm (recon) 
        Ref: Refernce fully sampled image, against which performance of the output should be validated
        Undersampled: Input undersampled image 
        path4output: Path for storing the output
        convertedTo2D: Whether it images were converted to 2D
        getSSIMMap: Whether like to save SSIM Map
        """

        # Creating all the necesaary file paths
        if not os.path.exists(path4output):
            os.makedirs(path4output)
        accuracyFileName = os.path.join(path4output,'accuracy.txt')
        accuracyCorrFileName = os.path.join(path4output,'afterDataConsistancy.txt')
        ssimMapFileName = os.path.join(path4output,'ssimmap.nii.gz')
        ssimMapCorrFileName = os.path.join(path4output,'ssimmapCorrected.nii.gz')
        fullySampledVolFileName = os.path.join(path4output,'fully.nii.gz')
        underSampledVolFileName = os.path.join(path4output,'under.nii.gz')
        outputVolFileName = os.path.join(path4output,'recon.nii.gz')
        outputCorrVolFileName = os.path.join(path4output,'reconCorrected.nii.gz')
        fullySampledCoilVolFileName = os.path.join(path4output,'fully_coil.nii.gz')
        underSampledCoilVolFileName = os.path.join(path4output,'under_coil.nii.gz')
        outputCoilVolFileName = os.path.join(path4output,'recon_coil.nii.gz')
        outputCorrCoilVolFileName = os.path.join(path4output,'reconCorrected_coil.nii.gz')
        underDiffSampledVolFileName = os.path.join(path4output,'diff_under.nii.gz')
        outputDiffVolFileName = os.path.join(path4output,'diff_recon.nii.gz')
        outputDiffCorrVolFileName = os.path.join(path4output,'diff_reconCorrected.nii.gz')
        ssimMapUndersampledFileName = os.path.join(path4output,'ssimmapOfUndersampled.nii.gz')
        accuracyUndersampledFileName = os.path.join(path4output,'accuracyOfUndersampled.txt')
        improvementFileName = os.path.join(path4output,'improvement.txt')
        improvementCorrFileName = os.path.join(path4output,'improvementCorrected.txt')
        additonalData = os.path.join(path4output,'dataDict.pkl')

        if mask is not None:
            dataDict = {
                "mask": mask.cpu().numpy() if type(mask) is torch.Tensor else mask
            }
        else:
            dataDict = {}

        #Do domain transformations for non-Image domains
        if self.domain == 'fourier':
            if self.use_data_consistency:
                OutK = Out
                RefK = Ref
                UndersampledK = Undersampled
            Out = torch.abs(ifft2c(Out))
            Ref = torch.abs(ifft2c(Ref))
            Undersampled = torch.abs(ifft2c(Undersampled))
        elif self.domain == 'hartley':
            sys.exit("save_reconds: hartley domain not implimented")  
        elif self.domain == 'image' or self.domain == 'compleximage': 
            if self.use_data_consistency:
                OutK = fft2c(Out) if OutK is None else OutK.to(Out.device)
                RefK = fft2c(Ref) if RefK is None else RefK.to(Ref.device)
                UndersampledK = None if UndersampledK is None else UndersampledK.to(Undersampled.device) #as we don't have any real undersampled K if we are just having image data (if UndersampledK isn't supplied additionally)
            if self.domain == 'compleximage':
                Out = torch.abs(Out)
                Ref = torch.abs(Ref)
                Undersampled = torch.abs(Undersampled)
        else:
            sys.exit("save_reconds: domain not defined")     

        if self.save_raw:
            dataDict = {
                **dataDict,
                "OutK": OutK.cpu().numpy() if type(OutK) is torch.Tensor else OutK,
                "RefK": RefK.cpu().numpy() if type(RefK) is torch.Tensor else RefK,
                "UndersampledK": UndersampledK.cpu().numpy() if type(UndersampledK) is torch.Tensor else UndersampledK
            }

        Out = Out.cpu().numpy().astype('float32') if type(Out) is torch.Tensor else Out.astype('float32')
        Ref = Ref.cpu().numpy().astype('float32') if type(Ref) is torch.Tensor else Ref.astype('float32')
        Undersampled = Undersampled.cpu().numpy().astype('float32') if type(Undersampled) is torch.Tensor else Undersampled.astype('float32')

        if self.channelSum:
            if self.save_coil_imgs:
                FileSave(np.moveaxis(Ref, 0, -1), fullySampledCoilVolFileName)
                FileSave(np.moveaxis(Undersampled, 0, -1), underSampledCoilVolFileName)
                FileSave(np.moveaxis(Out, 0, -1), outputCoilVolFileName)
            Out = root_sum_of_squares_np(Out)
            Ref = root_sum_of_squares_np(Ref)
            Undersampled = root_sum_of_squares_np(Undersampled)

        #TODO: implement
        # if roi_crop_res is not None:
        #     Out = transforms.center_crop(Out, roi_crop_res)
        #     Ref = transforms.center_crop(Ref, roi_crop_res)
        #     Undersampled = transforms.center_crop(Undersampled, roi_crop_res)

        FileSave(Ref, fullySampledVolFileName)
        FileSave(Undersampled, underSampledVolFileName)
        FileSave(Out, outputVolFileName)
        
        if self.doComparisons:
            if self.convertTo2D:
                Out = Nifti3Dto2D(Out)
                Ref = Nifti3Dto2D(Ref)
                Undersampled = Nifti3Dto2D(Undersampled)

            #Calculate Accuracy of the Output
            errorPercent = np.mean(Out != Ref) * 100
            mse = ((Out - Ref) ** 2).mean()

            #Calculate Accuracy of the Undersampled Image
            errorPercentUndersampled = np.mean(Undersampled != Ref) * 100
            mseUndersampled = ((Undersampled - Ref) ** 2).mean()

            rmse = sqrt(mse)
            rmseUndersampled = sqrt(mseUndersampled)

            #Calculate SSIM of Output and save SSIM as well as aaccuracy
            if self.getSSIMMap: #Data Range TODO, should be 1 but it was max-min of output Out4SSIM.max() - Out4SSIM.min()
                ssim, ssimMAP = compare_ssim(Ref, Out, data_range=1, multichannel=False, full=True) #with Map 
                FileSave(ssimMAP, ssimMapFileName)
            else:
                ssim = compare_ssim(Ref, Out, data_range=1, multichannel=False) #without Map

            psnr = peak_signal_noise_ratio(Out, Ref, data_range=1)

            dif_out = Ref - Out
            FileSave(dif_out, outputDiffVolFileName)
            dif_out_std = np.std(dif_out)
        
            file = open(accuracyFileName,"w")
            file.write("Error Percent: " + str(errorPercent)) 
            file.write("\r\nMSE: " + str(mse)) 
            file.write("\r\nRMSE: " + str(rmse)) 
            file.write("\r\nMean SSIM: " + str(ssim))  
            file.write("\r\nPSNR: " + str(psnr))  
            file.write("\r\nSDofDiff: " + str(dif_out_std))  
            file.close() 
            
            #Calculate SSIM of Undersampled and save SSIM as well as aaccuracy
            if self.getSSIMMap: #Data Range TODO, should be 1 but it was max-min of under Under4SSIM.max() - Under4SSIM.min()
                ssimUndersampled, ssimUndersampledMAP = compare_ssim(Ref, Undersampled, data_range=1, multichannel=False, full=True) #3D, with Map 
                FileSave(ssimUndersampledMAP, ssimMapUndersampledFileName)
            else:
                ssimUndersampled = compare_ssim(Ref, Undersampled, data_range=1, multichannel=False) #3D 
            
            psnrUndersampled = peak_signal_noise_ratio(Undersampled, Ref, data_range=1)

            dif_inp = Ref - Undersampled
            FileSave(dif_inp, underDiffSampledVolFileName)
            dif_inp_std = np.std(dif_inp)

            file = open(accuracyUndersampledFileName,"w")
            file.write("Error Percent: " + str(errorPercentUndersampled)) 
            file.write("\r\nMSE: " + str(mseUndersampled)) 
            file.write("\r\nRMSE: " + str(rmseUndersampled)) 
            file.write("\r\nMean SSIM: " + str(ssimUndersampled))  
            file.write("\r\nPSNR: " + str(psnrUndersampled))  
            file.write("\r\nSDofDiff: " + str(dif_inp_std))  
            file.close() 
        
            #Check for Accuracy Improvement
            errorPercentImprovement = errorPercentUndersampled - errorPercent
            mseImprovement = mseUndersampled - mse
            rmseImprovement = rmseUndersampled - rmse
            ssimImprovement = ssim - ssimUndersampled
            psnrImprovement = psnr - psnrUndersampled
            diffSDImprovement = dif_inp_std - dif_out_std
            file = open(improvementFileName,"w")
            file.write("Error Percent: " + str(errorPercentImprovement)) 
            file.write("\r\nMSE: " + str(mseImprovement)) 
            file.write("\r\nRMSE: " + str(rmseImprovement)) 
            file.write("\r\nMean SSIM: " + str(ssimImprovement)) 
            file.write("\r\nPSNR: " + str(psnrImprovement)) 
            file.write("\r\nSDofDiff: " + str(diffSDImprovement))  
            file.close() 

        if self.use_data_consistency:
            OutK_corrected = self.datacon.apply(OutK, RefK, UndersampledK, mask)
            if self.save_raw:
                dataDict = {
                    **dataDict,
                    "OutK_corrected": OutK_corrected.cpu().numpy() if type(OutK_corrected) is torch.Tensor else OutK_corrected
                }
            Out_corrected = torch.abs(ifft2c(OutK_corrected))
            Out_corrected = Out_corrected.cpu().numpy().astype('float32') if type(Out_corrected) is torch.Tensor else Out_corrected.astype('float32')
            if self.channelSum:
                if self.save_coil_imgs:
                    FileSave(np.moveaxis(Out_corrected, 0, -1), outputCorrCoilVolFileName)
                Out_corrected = root_sum_of_squares_np(Out_corrected)            
            FileSave(Out_corrected, outputCorrVolFileName)

            if self.doComparisons:
                if self.convertTo2D:
                    Out_corrected = Nifti3Dto2D(Out_corrected)

                #Calculate Accuracy of the Output
                errorPercent = np.mean(Out_corrected != Ref) * 100
                mse = ((Out_corrected - Ref) ** 2).mean()
                rmse = sqrt(mse)

                #Calculate SSIM of Output and save SSIM as well as aaccuracy
                if self.getSSIMMap: #Data Range TODO, should be 1 but it was max-min of output Out4SSIM.max() - Out4SSIM.min()
                    ssim, ssimMAP = compare_ssim(Ref, Out_corrected, data_range=1, multichannel=False, full=True) #with Map 
                    FileSave(ssimMAP, ssimMapCorrFileName)
                else:
                    ssim = compare_ssim(Ref, Out_corrected, data_range=1, multichannel=False) #without Map

                psnr = peak_signal_noise_ratio(Out_corrected, Ref, data_range=1)

                dif_out = Ref - Out_corrected
                FileSave(dif_out, outputDiffCorrVolFileName)
                dif_out_std = np.std(dif_out)
            
                file = open(accuracyCorrFileName,"w")
                file.write("Error Percent: " + str(errorPercent)) 
                file.write("\r\nMSE: " + str(mse)) 
                file.write("\r\nRMSE: " + str(rmse)) 
                file.write("\r\nMean SSIM: " + str(ssim))  
                file.write("\r\nPSNR: " + str(psnr))  
                file.write("\r\nSDofDiff: " + str(dif_out_std))  
                file.close() 

                #Check for Accuracy Improvement
                errorPercentImprovement = errorPercentUndersampled - errorPercent
                mseImprovement = mseUndersampled - mse
                rmseImprovement = rmseUndersampled - rmse
                ssimImprovement = ssim - ssimUndersampled
                psnrImprovement = psnr - psnrUndersampled
                diffSDImprovement = dif_inp_std - dif_out_std
                file = open(improvementCorrFileName,"w")
                file.write("Error Percent: " + str(errorPercentImprovement)) 
                file.write("\r\nMSE: " + str(mseImprovement)) 
                file.write("\r\nRMSE: " + str(rmseImprovement)) 
                file.write("\r\nMean SSIM: " + str(ssimImprovement)) 
                file.write("\r\nPSNR: " + str(psnrImprovement)) 
                file.write("\r\nSDofDiff: " + str(diffSDImprovement))  
                file.close() 

        if bool(additonalData):
            with open(additonalData, 'wb') as handle:
                pickle.dump(dataDict, handle)



