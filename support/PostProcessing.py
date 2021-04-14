class PostProcessing(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def ssd(img1, img2):
        return np.sum( (img1 - img2) ** 2 )

    def MinMaxNorm(data):
        return (data - data.min()) / (data.max() - data.min())

    def CalcualteNSave(fully, recon, recon_improved, fullpath_subfolder, isNorm):
        ssim_recon = ssim(fully, recon)
        ssim_recon_improved = ssim(fully, recon_improved)
        #mse_recon_improved = mse(fully, recon_improved)
        #nrmse_recon_improved = nrmse(fully, recon_improved)
        psnr_recon_improved = psnr(fully, recon_improved)
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


