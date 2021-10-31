import os
from glob import glob
from utils.HandleDicom import FolderRead
from utils.HandleNifti import FileSave
import cv2

root = r'E:\Datasets\ADNI\MCI_SMC_T1_10'
output_root = r'E:\Datasets\ADNI\NIFTIs\MP-RAGE'
meas_name = 'MP-RAGE'

root = os.path.join(root, 'ADNI')
subject_list = [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI))]

for subject in subject_list:
    meas_root = os.path.join(root, subject, meas_name)
    save_root = os.path.join(output_root, subject)
    if os.path.isdir(meas_root):
        scans = [dI for dI in os.listdir(meas_root) if os.path.isdir(os.path.join(meas_root,dI))]
        for scan in scans:
            scan_root = os.path.join(meas_root, scan)
            acqs = [dI for dI in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root,dI))]
            for acq in acqs:
                dicom_root = os.path.join(scan_root, acq)
                print(dicom_root)
                vol = FolderRead(dicom_root).squeeze()
                if vol.shape[0] != vol.shape[1]:
                    continue
                try:
                    for sl in range(vol.shape[2]):          
                        ##For OASIS Original Orientation
                        #vol[...,sl] = cv2.flip(vol[...,sl], 0) #Horizontal Flip. For Vertical the second param will be 1 - to reach same as OASIS original DS
                    
                        #For OASIS Rotated Orientation (which is used in NN)
                        vol[...,sl] = cv2.flip(vol[...,sl], 1)
                        vol[...,sl] = cv2.transpose(vol[...,sl])
                    os.makedirs(save_root, exist_ok=True)
                    file_name = scan + '_' + acq + '.nii.gz'
                    FileSave(vol.squeeze(), os.path.join(save_root, file_name))
                except Exception as ex:
                    print(ex)
