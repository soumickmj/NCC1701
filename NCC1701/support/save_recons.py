#!/usr/bin/env python

"""
This module helps to validate and store after reconsturcting.

This code is created with the help of pytorch_modelsize by jacobkimmel 
link: https://github.com/jacobkimmel/pytorch_modelsize

"""

import os
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as compare_ssim
from utils.HandleNifti import FileSave, Nifti3Dto2D, Nifti2Dto3D

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Nearly Finished"


def ValidateNStore(Out,Ref,Undersampled,path4output,OutNoClear=None,UndersampledNoClear=None,convertTo2D=False,getSSIMMap=False,channelSum=True,doComparisons=True):
    """
    This is a common function, called by the two following functions (ValidateNStoreRecons & ValidateNStoreReconsROI), not to be called directly
    This function validates the output againest a reference image (fully sampled), also checks how much the image have improved from the input image (undersampled)
    Parameters:
    Out: Output of the algorithm (recon) [In case of ValidateNStoreReconsROI, this is OutClear (only ROI)]
    Ref: Refernce fully sampled image, against which performance of the output should be validated
    Undersampled: Input undersampled image [In case of ValidateNStoreReconsROI, this is UndersampledClear (only ROI)]
    path4output: Path for storing the output
    OutNoClear: Output of the algorithm (recon), not just ROI but complete before ROICleared Image [In case of ValidateNStoreReconsROI only, or else None]
    UndersampledNoClear: Input undersampled image, not just ROI but complete before ROICleared Image [In case of ValidateNStoreReconsROI only, or else None]
    convertedTo2D: Whether it images were converted to 2D
    getSSIMMap: Whether like to save SSIM Map
    """

    # Creating all the necesaary file names
    if not os.path.exists(path4output):
        os.makedirs(path4output)
    accuracyFileName = os.path.join(path4output,'accuracy.txt')
    ssimMapFileName = os.path.join(path4output,'ssimmap.mat')
    fullySampledVolFileName = os.path.join(path4output,'fully.nii')
    underSampledVolFileName = os.path.join(path4output,'under.nii')
    outputVolFileName = os.path.join(path4output,'recon.nii')
    outputClearVolFileName = os.path.join(path4output,'reconClear.nii')
    ssimMapUndersampledFileName = os.path.join(path4output,'ssimmapOfUndersampled.mat')
    accuracyUndersampledFileName = os.path.join(path4output,'accuracyOfUndersampled.txt')
    improvementFileName = os.path.join(path4output,'improvement.txt')

    if(channelSum):
         Out = np.sum(Out,axis=-1)
         Ref = np.sum(Ref,axis=-1)
         Undersampled = np.sum(Undersampled,axis=-1)
         if(OutNoClear is not None):
             OutNoClear = np.sum(OutNoClear,axis=-1)
             UndersampledNoClear = np.sum(UndersampledNoClear,axis=-1)

    #If OutNoClear is not None, that means 'ROI Clear' was performed earlier. So, save files accordinaly.
    #IF ROI Clear was performed, then two additional params OutNoClear and UndersampledNoClear must have been supplied to the algorithm
    if(OutNoClear is not None):
        OutNoClear = Nifti2Dto3D(OutNoClear)
        UndersampledNoClear = Nifti2Dto3D(UndersampledNoClear)
        FileSave(Ref, fullySampledVolFileName)
        FileSave(UndersampledNoClear, underSampledVolFileName)
        FileSave(Out, outputClearVolFileName)
        FileSave(OutNoClear, outputVolFileName)
    else:
        FileSave(Ref, fullySampledVolFileName)
        FileSave(Undersampled, underSampledVolFileName)
        FileSave(Out, outputVolFileName)
    
    if doComparisons:
        if(convertTo2D):
            Out = Nifti3Dto2D(Out)
            Ref = Nifti3Dto2D(Ref)
            Undersampled = Nifti3Dto2D(Undersampled)

        #Calculate Accuracy of the Output
        errorPercent = np.mean(Out != Ref) * 100
        mse = ((Out - Ref) ** 2).mean()

        #Calculate Accuracy of the Undersampled Image
        errorPercentUndersampled = np.mean(Undersampled != Ref) * 100
        mseUndersampled = ((Undersampled - Ref) ** 2).mean()


        if(np.iscomplex(Out).any()):
            rmse = sqrt(abs(mse))
            rmseUndersampled = sqrt(abs(mseUndersampled))
            Ref4SSIM = abs(Ref).astype('float32')
            Out4SSIM = abs(Out).astype('float32')
            Under4SSIM = abs(Undersampled).astype('float32')
        else:
            rmse = sqrt(mse)
            rmseUndersampled = sqrt(mseUndersampled)
            Ref4SSIM = Ref.astype('float32')
            Out4SSIM = Out.astype('float32')
            Under4SSIM = Undersampled.astype('float32')

        #Calculate SSIM of Output and save SSIM as well as aaccuracy
        if(getSSIMMap):
            ssim, ssimMAP = compare_ssim(Ref4SSIM, Out4SSIM, data_range=Out4SSIM.max() - Out4SSIM.min(), multichannel=True, full=True) #with Map 
            sio.savemat(ssimMapFileName, {'ssimMAP':ssimMAP})
        else:
            ssim = compare_ssim(Ref4SSIM, Out4SSIM, data_range=Out4SSIM.max() - Out4SSIM.min(), multichannel=True) #without Map
    
        file = open(accuracyFileName,"w")
        file.write("Error Percent: " + str(errorPercent)) 
        file.write("\r\nMSE: " + str(mse)) 
        file.write("\r\nRMSE: " + str(rmse)) 
        file.write("\r\nMean SSIM: " + str(ssim))  
        file.close() 
        
        #Calculate SSIM of Undersampled and save SSIM as well as aaccuracy
        if(getSSIMMap):
            ssimUndersampled, ssimUndersampledMAP = compare_ssim(Ref4SSIM, Under4SSIM, data_range=Under4SSIM.max() - Under4SSIM.min(), multichannel=True, full=True) #3D, with Map 
            sio.savemat(ssimMapUndersampledFileName, {'ssimMAP':ssimUndersampledMAP})
        else:
            ssimUndersampled = compare_ssim(Ref4SSIM, Under4SSIM, data_range=Under4SSIM.max() - Under4SSIM.min(), multichannel=True) #3D 
           
        file = open(accuracyUndersampledFileName,"w")
        file.write("Error Percent: " + str(errorPercentUndersampled)) 
        file.write("\r\nMSE: " + str(mseUndersampled)) 
        file.write("\r\nRMSE: " + str(rmseUndersampled)) 
        file.write("\r\nMean SSIM: " + str(ssimUndersampled))  
        file.close() 
    
        #Check for Accuracy Improvement
        errorPercentImprovement = errorPercentUndersampled - errorPercent
        mseImprovement = mseUndersampled - mse
        rmseImprovement = rmseUndersampled - rmse
        ssimImprovement = ssim - ssimUndersampled
        file = open(improvementFileName,"w")
        file.write("Error Percent: " + str(errorPercentImprovement)) 
        file.write("\r\nMSE: " + str(mseImprovement)) 
        file.write("\r\nRMSE: " + str(rmseImprovement)) 
        file.write("\r\nMean SSIM: " + str(ssimImprovement))  
        file.close() 
    
        #If the images were convereted to 2D earlier, convert them back to 3D
        if(convertTo2D):
            Ref = Nifti2Dto3D(Ref)
            Undersampled = Nifti2Dto3D(Undersampled)
            Out = Nifti2Dto3D(Out)

    

def ValidateNStoreRecons(Out,Ref,Undersampled,path4output,convertTo2D=False,getSSIMMap=False,channelSum=True,doComparisons=True):
    """
    This is a common function, called by the two following functions (ValidateNStoreRecons & ValidateNStoreReconsROI), not to be called directly
    This function validates the output againest a reference image (fully sampled), also checks how much the image have improved from the input image (undersampled)
    Parameters:
    Out: Output of the algorithm (recon) 
    Ref: Refernce fully sampled image, against which performance of the output should be validated
    Undersampled: Input undersampled image 
    path4output: Path for storing the output
    convertTo2D: Whether to convert images to 2D
    getSSIMMap: Whether like to save SSIM Map
    """

    #Call the common validation fuction
    ValidateNStore(Out,Ref,Undersampled,path4output,convertTo2D=convertTo2D,getSSIMMap=getSSIMMap,channelSum=channelSum,doComparisons=doComparisons)
    
#def ValidateNStoreReconsROI(Out,Ref,Undersampled,path4output,getSSIMMap=False):
#    """
#    This is a common function, called by the two following functions (ValidateNStoreRecons & ValidateNStoreReconsROI), not to be called directly
#    This function validates the output againest a reference image (fully sampled), also checks how much the image have improved from the input image (undersampled)
#    Parameters:
#    Out: Output of the algorithm (recon) 
#    Ref: Refernce fully sampled image, against which performance of the output should be validated
#    Undersampled: Input undersampled image 
#    path4output: Path for storing the output
#    getSSIMMap: Whether like to save SSIM Map
#    """

#    #Code for clearing out the NonROI
#    from utils.GetROI import ClearNonROIFromUndersampled

#    #Try to convert it to 2D, but it might so happen that they are already in 2D
#    try:
#        Out = Nifti3Dto2D(Out)
#        Ref = Nifti3Dto2D(Ref)
#        Undersampled = Nifti3Dto2D(Undersampled)
#    except:
#        print("Couldn't convert it to 2D from 3D. Is it already in 2D?")

#    #Clear Non-ROI 
#    OutClear = ClearNonROIFromUndersampled(Ref,Out)
#    UndersampledClear = ClearNonROIFromUndersampled(Ref,Undersampled)
    
#    #Call the common validation fuction
#    ValidateNStore(OutClear,Ref,UndersampledClear,path4output,convertedTo2D=True,getSSIMMap=getSSIMMap)