import os
import xlsxwriter
from utils.HandleNifti import FileRead2D, FileSave
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
import math

def quality_check(img1, img2):
    mseValue = mse(img1, img2)
    ssdValue = np.sum( (img1 - img2) ** 2 )
    nrmseValue = nrmse(img1, img2)
    psnrValue = psnr(img1, img2)
    return psnrValue, mseValue, nrmseValue, ssdValue

def MultisliceCombiner(root_path_input, root_path_output, isFolderInFolder=True, isAnyCorrected=False):
    folders = [f for f in os.listdir(root_path_input) if f != 'checkpoints' and not os.path.isfile(os.path.join(root_path_input, f))] #Get all subfolders (subject specific folders) from root
    print(root_path_input)
    for folder in folders:   
        print(folder)
        fullpath_folder = os.path.join(root_path_input, folder)
        fullpath_folder_output = os.path.join(root_path_output, folder)
        subfolders = [f for f in os.listdir(fullpath_folder) if not os.path.isfile(os.path.join(fullpath_folder, f))] #Get all subfolders (subject specific folders) from root
        subfolder_output = '_'.join(subfolders[0].split('_')[:-1])
        fullpath_subfolder_output = os.path.join(fullpath_folder_output, subfolder_output)
        os.makedirs(fullpath_subfolder_output, exist_ok = True)
        fully = []
        recon = []
        reconCorrected = []
        reconCorrectedNorm = []
        under = []
        for subfolder in subfolders:
            fullpath_subfolder = os.path.join(fullpath_folder, subfolder)
            fullySlc = FileRead2D(os.path.join(fullpath_subfolder, 'fully.nii'))
            fully.append(fullySlc.squeeze())
            reconSlc = FileRead2D(os.path.join(fullpath_subfolder, 'recon.nii'))
            recon.append(reconSlc.squeeze())
            try:
                reconCorrectedSlc = FileRead2D(os.path.join(fullpath_subfolder, 'reconCorrected.nii'))
                reconCorrected.append(reconCorrectedSlc.squeeze())
            except:
                pass
            try:
                reconCorrectedNormSlc = FileRead2D(os.path.join(fullpath_subfolder, 'reconCorrectedNorm.nii'))
                reconCorrectedNorm.append(reconCorrectedNormSlc.squeeze())
            except:
                pass
            underSlc = FileRead2D(os.path.join(fullpath_subfolder, 'under.nii'))
            under.append(underSlc.squeeze())
        fully = np.array(fully).transpose((1,2,0))
        FileSave(fully, os.path.join(fullpath_subfolder_output, 'fully.nii'))
        recon = np.array(recon).transpose((1,2,0))
        FileSave(recon, os.path.join(fullpath_subfolder_output, 'recon.nii'))
        ssim_fully_recon = ssim(fully.squeeze(), recon.squeeze())
        psnr_fully_recon, mse_fully_recon, rmse_fully_recon, ssd_fully_recon = quality_check(fully, recon)
        if(len(reconCorrected) > 0):
            reconCorrected = np.array(reconCorrected).transpose((1,2,0))
            FileSave(reconCorrected, os.path.join(fullpath_subfolder_output, 'reconCorrected.nii'))
            ssim_fully_reconCorrected = ssim(fully.squeeze(), reconCorrected.squeeze())
            psnr_fully_reconCorrected, mse_fully_reconCorrected, rmse_fully_reconCorrected, ssd_fully_reconCorrected = quality_check(fully, reconCorrected)
        if(len(reconCorrectedNorm) > 0):
            reconCorrectedNorm = np.array(reconCorrectedNorm).transpose((1,2,0))
            FileSave(reconCorrectedNorm, os.path.join(fullpath_subfolder_output, 'reconCorrectedNorm.nii'))
            ssim_fully_reconCorrectedNorm = ssim(fully.squeeze(), reconCorrectedNorm.squeeze())
            psnr_fully_reconCorrectedNorm, mse_fully_reconCorrectedNorm, rmse_fully_reconCorrectedNorm, ssd_fully_reconCorrectedNorm = quality_check(fully, reconCorrectedNorm)
        under = np.array(under).transpose((1,2,0))
        FileSave(under, os.path.join(fullpath_subfolder_output, 'under.nii'))
        ssim_fully_under = ssim(fully.squeeze(), under.squeeze())
        psnr_fully_under, mse_fully_under, rmse_fully_under, ssd_fully_under  = quality_check(fully, under)

        accuracy = os.path.join(fullpath_subfolder_output, 'accuracy.txt')   
        accuracyOfUndersampled = os.path.join(fullpath_subfolder_output, 'accuracyOfUndersampled.txt')          
        improvement = os.path.join(fullpath_subfolder_output, 'improvement.txt')  
        afterDataConsistancy = os.path.join(fullpath_subfolder_output, 'afterDataConsistancy.txt') 
        afterDataConsistancyNorm = os.path.join(fullpath_subfolder_output, 'afterDataConsistancyNorm.txt')

        file = open(accuracy,"w") 
        file.write("Error Percent: ") 
        file.write("\r\nMSE: " + str(mse_fully_recon)) 
        file.write("\r\nNRMSE: " + str(rmse_fully_recon)) 
        file.write("\r\nMean SSIM: " + str(ssim_fully_recon))  
        file.write("\r\nPSNR: " + str(psnr_fully_recon))  
        file.write("\r\nSSD: " + str(ssd_fully_recon))  
        file.close() 

        file = open(accuracyOfUndersampled,"w") 
        file.write("Error Percent: ") 
        file.write("\r\nMSE: " + str(mse_fully_under)) 
        file.write("\r\nNRMSE: " + str(rmse_fully_under)) 
        file.write("\r\nMean SSIM: " + str(ssim_fully_under))  
        file.write("\r\nPSNR: " + str(psnr_fully_under))  
        file.write("\r\nSSD: " + str(ssd_fully_under))  
        file.close() 

        file = open(improvement,"w") 
        file.write("Error Percent: ") 
        file.write("\r\nMSE: " + str(mse_fully_under - mse_fully_recon)) 
        file.write("\r\nNRMSE: " + str(rmse_fully_under-rmse_fully_recon)) 
        file.write("\r\nMean SSIM: " + str(ssim_fully_recon-ssim_fully_under))  
        file.write("\r\nPSNR: " + str(psnr_fully_recon-psnr_fully_under))  
        file.write("\r\nSSD: " + str(ssd_fully_under-ssd_fully_recon))  
        file.close() 

        if(len(reconCorrected) > 0):
            file = open(afterDataConsistancy,"w") 
            file.write("OldSSIM: " + str(ssim_fully_recon)) 
            file.write("\r\nNewSIIM: " + str(ssim_fully_reconCorrected)) 
            file.write("\r\nPSNR: " + str(psnr_fully_reconCorrected))  
            file.write("\r\nSSD: " + str(ssd_fully_reconCorrected))  
            file.close() 

        if(len(reconCorrectedNorm) > 0):
            file = open(afterDataConsistancyNorm,"w") 
            file.write("OldSSIM: " + str(ssim_fully_recon)) 
            file.write("\r\nNewSIIM: " + str(ssim_fully_reconCorrectedNorm)) 
            file.write("\r\nPSNR: " + str(psnr_fully_reconCorrectedNorm))  
            file.write("\r\nSSD: " + str(ssd_fully_reconCorrectedNorm))  
            file.close() 
    


#MultisliceCombiner(r'D:\Output\Qadro\Gen2\Attempt20_Redo-WithValidate-Radial60GA-Resnet2Dv2SSIMLoss-51to80SingleEach\singleslice', 
#                   r'D:\Output\Attempt37-WithValidate-OASIS-Uniform28-4step_Fresh-Resnet2Dv2SSIMLoss-51to80SingleEach\combined', True, True)
