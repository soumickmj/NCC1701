import os
from utils.HandleNifti import FileRead2D, FileSave
import scipy.io as sio
import numpy as np
from Math.FrequencyTransforms import ifft2c, fft2c
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
from sklearn.preprocessing import MinMaxScaler
import math

import scipy.misc
from pynufft import NUFFT as NUFFT_cpu
import pkg_resources

import matplotlib.pyplot as plt

def ssd(img1, img2):
    return np.sum( (img1 - img2) ** 2 )

def MinMaxNorm(data):
    try:
        return (data - data.min()) / (data.max() - data.min())
    except:
        try:
            return data / data.max()
        except:
            return data

def CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, isNorm):
    ssim_recon = ssim(fully, recon)
    ssim_recon_improved = ssim(fully, recon_improved)
    #mse_recon_improved = mse(fully, recon_improved)
    #nrmse_recon_improved = nrmse(fully, recon_improved)
    try:
        psnr_recon_improved = psnr(fully, recon_improved)
    except:
        psnr_recon_improved = -1
    ssd_recon_improved = ssd(fully, recon_improved)
    if(isNorm):
        FileSave(recon_improved, os.path.join(fullpath_subfolder, 'reconCorrectedNorm.nii'))
        file = open(os.path.join(fullpath_subfolder, 'afterDataConsistancyNorm.txt'),"w") 
    else:
        FileSave(recon_improved, os.path.join(fullpath_subfolder, 'reconCorrected.nii'))
        file = open(os.path.join(fullpath_subfolder, 'afterDataConsistancy.txt'),"w") 
    file.write("OldSSIM: " + str(ssim_recon)) 
    file.write("\r\nNewSIIM: " + str(ssim_recon_improved)) 
    file.write("\r\nPSNR: " + str(psnr_recon_improved))  
    file.write("\r\nSSD: " + str(ssd_recon_improved))  
    file.close() 

def ApplyDataConsis(root_path, isRadial, doNorm, mask_or_om_path, interpolationSize4NUFFT = 6, usCGSolver = False, imageSize = 256, updateStats = True):
    folders = [f for f in os.listdir(root_path) if f != 'checkpoints' and not os.path.isfile(os.path.join(root_path, f))] #Get all subfolders (subject specific folders) from root
    print(root_path)
    if(not isRadial):
        mask = sio.loadmat(mask_or_om_path)['mask'].astype(bool)
        undersamplingMask = mask.astype(int)
        missingMask = (~mask).astype(int)
    else:
        temp_mat =  sio.loadmat(mask_or_om_path)
        om = temp_mat['om']
        invom = temp_mat['invom']
        fullom = temp_mat['fullom']
        dcf = temp_mat['dcf'].squeeze()
        dcfFullRes = temp_mat['dcfFullRes'].squeeze()
        baseresolution = imageSize*2

        NufftObjOM = NUFFT_cpu()
        NufftObjInvOM = NUFFT_cpu()
        NufftObjFullOM = NUFFT_cpu()

        Nd = (baseresolution, baseresolution)  # image size
        Kd = (baseresolution, baseresolution)  # k-space size - TODO: multiply back by 2
        Jd = (interpolationSize4NUFFT, interpolationSize4NUFFT)  # interpolation size

        NufftObjOM.plan(om, Nd, Kd, Jd)
        NufftObjInvOM.plan(invom, Nd, Kd, Jd)
        NufftObjFullOM.plan(fullom, Nd, Kd, Jd)


    for folder in folders:   
        print(folder)
        fullpath_folder = os.path.join(root_path, folder)
        subfolders = [f for f in os.listdir(fullpath_folder) if not os.path.isfile(os.path.join(fullpath_folder, f))] #Get all subfolders (subject specific folders) from root
        for subfolder in subfolders:
            fullpath_subfolder = os.path.join(fullpath_folder, subfolder)
            fully = np.float64(FileRead2D(os.path.join(fullpath_subfolder, 'fully.nii')).squeeze())
            recon = np.float64(FileRead2D(os.path.join(fullpath_subfolder, 'recon.nii')).squeeze())

            if(updateStats):
                under = np.float64(FileRead2D(os.path.join(fullpath_subfolder, 'under.nii')).squeeze())
                accuracy = os.path.join(fullpath_subfolder, 'accuracyExtended.txt')   
                accuracyOfUndersampled = os.path.join(fullpath_subfolder, 'accuracyOfUndersampledExtended.txt')          
                improvement = os.path.join(fullpath_subfolder, 'improvementExtended.txt')  

                mseRecon = mse(fully, recon)
                try:
                    nrmseRecon = nrmse(fully, recon)
                except:
                    nrmseRecon = -1
                ssimRecon = ssim(fully, recon)
                try:
                    psnrRecon = psnr(fully, recon)
                except:
                    psnrRecon = -1
                ssdRecon = ssd(fully, recon)

                file = open(accuracy,"w") 
                file.write("Stats: ") 
                file.write("\r\nMSE: " + str(mseRecon)) 
                file.write("\r\nNRMSE: " + str(nrmseRecon)) 
                file.write("\r\nMean SSIM: " + str(ssimRecon))  
                file.write("\r\nPSNR: " + str(psnrRecon))  
                file.write("\r\nSSD: " + str(ssdRecon))  
                file.close() 

               
                mseUnder = mse(fully, under)
                #mseUnder = -1
                try:
                    nrmseUnder = nrmse(fully, under)
                except:
                    nrmseUnder = -1
                ssimUnder = ssim(fully, under)
                try:
                    psnrUnder = psnr(fully, under)
                except:
                    psnrUnder = -1
                ssdUnder = ssd(fully, under)

                file = open(accuracyOfUndersampled,"w") 
                file.write("Stats: ") 
                file.write("\r\nMSE: " + str(mseUnder)) 
                file.write("\r\nNRMSE: " + str(nrmseUnder)) 
                file.write("\r\nMean SSIM: " + str(ssimUnder))  
                file.write("\r\nPSNR: " + str(psnrUnder))  
                file.write("\r\nSSD: " + str(ssdUnder))  
                file.close() 

                file = open(improvement,"w") 
                file.write("Stats: ") 
                file.write("\r\nMSE: " + str(mseUnder - mseRecon)) 
                file.write("\r\nNRMSE: " + str(nrmseUnder - nrmseRecon)) 
                file.write("\r\nMean SSIM: " + str(ssimRecon - ssimUnder))  
                file.write("\r\nPSNR: " + str(psnrRecon - psnrUnder))  
                file.write("\r\nSSD: " + str(ssdUnder - ssdRecon))  
                file.close() 

            if(not isRadial):
                fully_k = fft2c(fully)
                fully_masked_k = fully_k * undersamplingMask

                recon_k = fft2c(recon)
                recon_masked_k = recon_k * missingMask

                recon_improved_k = fully_masked_k + recon_masked_k
                recon_improved = ifft2c(recon_improved_k)
                recon_improved = recon_improved.real.astype(fully.dtype)
                #recon_improved = np.abs(recon_improved)
                CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, False)

                if(doNorm):
                    fully = MinMaxNorm(fully)
                    recon = MinMaxNorm(recon)
                    #scaler = MinMaxScaler()
                    #fully = scaler.fit_transform(fully)
                    #recon = scaler.transform(recon)

                    fully_k = fft2c(fully)
                    fully_masked_k = fully_k * undersamplingMask

                    recon_k = fft2c(recon)
                    recon_masked_k = recon_k * missingMask

                    recon_improved_k = fully_masked_k + recon_masked_k
                    recon_improved = ifft2c(recon_improved_k)
                    recon_improved = recon_improved.real.astype(fully.dtype)
                    #recon_improved = np.abs(recon_improved)
                    CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, True)
            else:
                oversam_fully = np.zeros((baseresolution,baseresolution), dtype=fully.dtype)
                oversam_fully[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = fully

                oversam_recon = np.zeros((baseresolution,baseresolution), dtype=recon.dtype)
                oversam_recon[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = recon

                yUnder = NufftObjOM.forward(oversam_fully)
                yReconFull = NufftObjFullOM.forward(oversam_recon)
                #yMissing = yReconFull[len(yUnder):len(yUnder)+2]
                yMissing = NufftObjInvOM.forward(oversam_recon)

                yCorrected = np.concatenate((yUnder,yMissing))

                ##k2vec_ = NufftObjOM.k2vec(yReconFull)
                ##k2xx_ = NufftObjOM.k2xx(yReconFull)
                ##k2y_ = NufftObjOM.k2y(yReconFull)
                
                #kS_OMus = NufftObjOM.k2y2k(yReconFull) #returns k-space - following OM
                #kS_OMfs = NufftObjFullOM.k2y2k(yReconFull) #returns k-space - following full OM
                #kS_OMmiss = NufftObjInvOM.k2y2k(yReconFull) #returns k-space - following inv OM

                #sio.savemat('test.mat',{'kS_OMus':kS_OMus, 'kS_OMfs':kS_OMfs, 'kS_OMmiss':kS_OMmiss })


                #vec2k_ = NufftObjOM.vec2k(oversam_fully) #returns image
                ##vec2y_ = NufftObjOM.vec2y(yReconFull)
                #y2vec_ = NufftObjOM.y2vec(yUnder)
           
                if(usCGSolver):
                    oversam_recon_improved = NufftObjFullOM.solve(yCorrected, solver='cg',maxiter=50)
                else:
                    yCorrected = np.multiply(dcfFullRes,yCorrected)
                    oversam_recon_improved = NufftObjFullOM.adjoint(yCorrected)

                recon_improved = oversam_recon_improved[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2]
                recon_improved = recon_improved.real.astype(fully.dtype)
                #recon_improved = np.abs(recon_improved)
                CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, False)
                if(doNorm):
                    fully = MinMaxNorm(fully)
                    recon = MinMaxNorm(recon)

                    oversam_fully = np.zeros((baseresolution,baseresolution), dtype=fully.dtype)
                    oversam_fully[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = fully

                    oversam_recon = np.zeros((baseresolution,baseresolution), dtype=recon.dtype)
                    oversam_recon[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = recon

                    yUnder = NufftObjOM.forward(oversam_fully)
                    #yReconFull = NufftObjFullOM.forward(oversam_recon)
                    #yMissing = yReconFull[len(yUnder):len(yUnder)+2]
                    yMissing = NufftObjInvOM.forward(oversam_recon)

                    yCorrected = np.concatenate((yUnder,yMissing))
                    if(usCGSolver):
                        oversam_recon_improved = NufftObjFullOM.solve(yCorrected, solver='cg',maxiter=50)
                    else:
                        yCorrected = np.multiply(dcfFullRes,yCorrected)
                        oversam_recon_improved = NufftObjFullOM.adjoint(yCorrected)

                    recon_improved = oversam_recon_improved[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2]
                    recon_improved = recon_improved.real.astype(fully.dtype)
                    #recon_improved = np.abs(recon_improved)
                    CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, True)


#For 1DVarden30Mask
#ApplyDataConsis(r'D:\ROugh\uni51-80', False, True, r'D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat')

#For 1DUniformMask4step28per
#ApplyDataConsis(r'D:\ROugh\uni51-80', False, True, r'D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DUniformMask4step28per.mat')

#For Radial60spGA
#ApplyDataConsis(r'D:\ROugh\uni51-80', True, True, r'D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\Radial60spGA.mat')

#For Radial30spGA
#ApplyDataConsis(r'D:\ROugh\uni51-80', True, True, r'D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\Radial30spGA.mat')

#ApplyDataConsis(r'D:\Output\Qadro\Gen2\Attempt21_Redo-WithValidate-Radial30GA-Resnet2Dv2SSIMLoss-51to80SingleEach', True, True, r'D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\Radial30spGA.mat')
