"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Name of the file been changed from utils.py to fastmri_utils.py to avoid conflict with the utils package of NCC1701
"""


import os
import json
import scipy.io as sio
import h5py

import utils.fastMRI.TorchDSTransforms as transforms
from utils.fastMRI import evaluate
from utils.HandleNifti import FileSave


def save_reconstructions_h5(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.
    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


##### Added by Soumick

def ValidateNStoreRecons(Out,Ref,Undersampled,mask,path4output,getSSIMMap=False,channelSum=True,domain='fourier',use_data_consistency=True,roi_crop_res=None,save_raw=False):
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
    domain: 
    use_data_consistency:
    roi_crop_res:
    """

    if Out.is_cuda:
        Out = Out.cpu()
        Ref = Ref.cpu()
        Undersampled = Undersampled.cpu()

     # Creating all the necesaary file names
    if not os.path.exists(path4output):
        os.makedirs(path4output)
    accuracyFileName = os.path.join(path4output,'accuracy.txt')
    afterDataConsistancyFileName = os.path.join(path4output,'afterDataConsistancy.txt')
    ssimMapFileName = os.path.join(path4output,'ssimmap.mat')
    fullySampledVolFileName = os.path.join(path4output,'fully.nii')
    underSampledVolFileName = os.path.join(path4output,'under.nii')
    outputVolFileName = os.path.join(path4output,'recon.nii')
    outputCorrectedVolFileName = os.path.join(path4output,'reconCorrected.nii')
    rawmatFileName = os.path.join(path4output,'raw.mat')
    ssimMapUndersampledFileName = os.path.join(path4output,'ssimmapOfUndersampled.mat')
    accuracyUndersampledFileName = os.path.join(path4output,'accuracyOfUndersampled.txt')
    improvementFileName = os.path.join(path4output,'improvement.txt')    

    #Lets convert everything to fourier space
    if domain == 'fourier':
        Out = transforms.channel_to_complex(Out)
        Ref = transforms.channel_to_complex(Ref)
        Undersampled = transforms.channel_to_complex(Undersampled)
    elif domain == 'hartley':
        Out = transforms.hartley_to_fourier(Out)
        Ref = transforms.hartley_to_fourier(Ref)
        Undersampled = transforms.hartley_to_fourier(Undersampled)
    elif domain == 'image':
        Out = transforms.rfft2(Out)
        Ref = transforms.rfft2(Ref)
        Undersampled = transforms.rfft2(Undersampled)
    elif domain == 'compleximage':
        Out = transforms.fft2(Out)
        Ref = transforms.fft2(Ref)
        Undersampled = transforms.fft2(Undersampled)
    else:
        import sys
        sys.exit()        

    if channelSum:
        Out_image = transforms.root_sum_of_squares(transforms.complex_abs(transforms.ifft2(Out))).numpy()
        Ref_image = transforms.root_sum_of_squares(transforms.complex_abs(transforms.ifft2(Ref))).numpy()
        Undersampled_image = transforms.root_sum_of_squares(transforms.complex_abs(transforms.ifft2(Undersampled))).numpy()
    else: 
        Out_image = transforms.complex_abs(transforms.ifft2(Out)).numpy()
        Ref_image = transforms.complex_abs(transforms.ifft2(Ref)).numpy()
        Undersampled_image = transforms.complex_abs(transforms.ifft2(Undersampled)).numpy()

    if roi_crop_res is not None:
        Out_image = transforms.center_crop(Out_image, roi_crop_res)
        Ref_image = transforms.center_crop(Ref_image, roi_crop_res)
        Undersampled_image = transforms.center_crop(Undersampled_image, roi_crop_res)


    #######Calculate and store Accuracy of the Output
    errorPercent = evaluate.error_percent(Ref_image, Out_image)
    mse = evaluate.mse(Ref_image, Out_image)
    nmse = evaluate.nmse(Ref_image, Out_image)
    psnr = evaluate.psnr(Ref_image, Out_image)
    ssd = evaluate.ssd(Ref_image, Out_image)
    #Calculate SSIM of Output and save SSIM as well as aaccuracy
    if(getSSIMMap):
        ssim, ssimMAP = evaluate.ssim(Ref_image, Out_image, return_map=True)
        sio.savemat(ssimMapFileName, {'ssimMAP':ssimMAP})
    else:
        ssim = evaluate.ssim(Ref_image, Out_image)
    file = open(accuracyFileName,"w")
    file.write("Error Percent: " + str(errorPercent)) 
    file.write("\r\nMSE: " + str(mse)) 
    file.write("\r\nNMSE: " + str(nmse)) 
    file.write("\r\nMean SSIM: " + str(ssim))  
    file.write("\r\nPSNR: " + str(psnr))  
    file.write("\r\nSSD: " + str(ssd))  
    file.close()

    ################Calculate Accuracy of the Undersampled Image
    errorPercentUnder = evaluate.error_percent(Ref_image, Undersampled_image)
    mseUnder = evaluate.mse(Ref_image, Undersampled_image)
    nmseUnder = evaluate.nmse(Ref_image, Undersampled_image)
    psnrUnder = evaluate.psnr(Ref_image, Undersampled_image)
    ssdUnder = evaluate.ssd(Ref_image, Undersampled_image)
    #Calculate SSIM of Output and save SSIM as well as aaccuracy
    if(getSSIMMap):
        ssimUnder, ssimMAPUnder = evaluate.ssim(Ref_image, Undersampled_image, return_map=True)
        sio.savemat(ssimMapUndersampledFileName, {'ssimMAP':ssimMAPUnder})
    else:
        ssimUnder = evaluate.ssim(Ref_image, Undersampled_image)
    file = open(accuracyUndersampledFileName,"w")
    file.write("Error Percent: " + str(errorPercentUnder)) 
    file.write("\r\nMSE: " + str(mseUnder)) 
    file.write("\r\nNMSE: " + str(nmseUnder)) 
    file.write("\r\nMean SSIM: " + str(ssimUnder))  
    file.write("\r\nPSNR: " + str(psnrUnder))  
    file.write("\r\nSSD: " + str(ssdUnder))  
    file.close() 

    ##########Check for Accuracy Improvement and store
    errorPercentImprovement = errorPercentUnder - errorPercent
    mseImprovement = mseUnder - mse
    nmseImprovement = nmseUnder - nmse
    ssimImprovement = ssim - ssimUnder
    psnrImprovement = psnr - psnrUnder
    ssdImprovement = ssdUnder - ssd
    file = open(improvementFileName,"w")
    file.write("Error Percent: " + str(errorPercentImprovement)) 
    file.write("\r\nMSE: " + str(mseImprovement)) 
    file.write("\r\nRMSE: " + str(nmseImprovement)) 
    file.write("\r\nMean SSIM: " + str(ssimImprovement))  
    file.write("\r\nPSNR: " + str(psnrImprovement))  
    file.write("\r\nSSD: " + str(ssdImprovement))  
    file.close()


    ##########Save images
    FileSave(Ref_image, fullySampledVolFileName)
    FileSave(Undersampled_image, underSampledVolFileName)
    FileSave(Out_image, outputVolFileName)

    if save_raw:
        raw_dict = {'Out': Out.numpy(), 'Ref': Ref.numpy(), 'Undersampled': Undersampled.numpy(), 'mask':mask.numpy()}

    if use_data_consistency:
        missing_mask = 1-mask
        missing_kSpace, _ = transforms.apply_mask(Out, mask=missing_mask)
        Out_corrected = Undersampled + missing_kSpace
        if channelSum:
            Out_corrected_image = transforms.root_sum_of_squares(transforms.complex_abs(transforms.ifft2(Out_corrected))).numpy()
        else: 
            Out_corrected_image = transforms.complex_abs(transforms.ifft2(Out_corrected)).numpy()

        if roi_crop_res is not None:
            Out_corrected_image = transforms.center_crop(Out_corrected_image, roi_crop_res)
        
        ################Calculate Accuracy of the Corrected Image
        errorPercent = evaluate.error_percent(Ref_image, Out_corrected_image)
        mse = evaluate.mse(Ref_image, Out_corrected_image)
        nmse = evaluate.nmse(Ref_image, Out_corrected_image)
        psnr = evaluate.psnr(Ref_image, Out_corrected_image)
        ssd = evaluate.ssd(Ref_image, Out_corrected_image)
        #Calculate SSIM of Output and save SSIM as well as aaccuracy
        if(getSSIMMap):
            ssimCorrected, ssimMAP = evaluate.ssim(Ref_image, Out_corrected_image, return_map=True)
            sio.savemat(ssimMapFileName, {'ssimMAP':ssimMAP})
        else:
            ssimCorrected = evaluate.ssim(Ref_image, Out_corrected_image)
        file = open(afterDataConsistancyFileName,"w")
        file.write("OldSSIM: " + str(ssim)) 
        file.write("\r\nNewSIIM: " + str(ssimCorrected)) 
        file.write("\r\nPSNR: " + str(psnr))  
        file.write("\r\nSSD: " + str(ssd))  
        file.write("\r\nError Percent: " + str(errorPercent))  
        file.write("\r\nMSE: " + str(mse))    
        file.write("\r\nNMSE: " + str(nmse))  
        file.close() 

        ##########Save corrected image
        FileSave(Out_corrected_image, outputCorrectedVolFileName)

        if save_raw:
            raw_dict['Out_corrected'] = Out_corrected.numpy()
            raw_dict['missing_kSpace'] = missing_kSpace.numpy()


    if save_raw:
        sio.savemat(rawmatFileName, raw_dict)