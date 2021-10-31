import os
from glob import glob
from utils.HandleDicom import FolderRead
from utils.HandleNifti import FileSave
import cv2

root = r'E:\Datasets\ResNetTestSet\Skyra\DICOM'
output_root = r'E:\Datasets\ResNetTestSet\Skyra\NIFTIs'
widthEqualsHeight = True

acq_list = [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI))]

for acq in acq_list:
    dicom_root = os.path.join(root, acq)
    save_root = os.path.join(output_root, acq)
    print(dicom_root)
    vol = FolderRead(dicom_root).squeeze()
    if widthEqualsHeight and vol.shape[0] != vol.shape[1]:
        continue
    try:
        for sl in range(vol.shape[2]):          
            ##For OASIS Original Orientation
            #vol[...,sl] = cv2.flip(vol[...,sl], 0) #Horizontal Flip. For Vertical the second param will be 1 - to reach same as OASIS original DS
                    
            #For OASIS Rotated Orientation (which is used in NN)
            vol[...,sl] = cv2.flip(vol[...,sl], 1)
            vol[...,sl] = cv2.transpose(vol[...,sl])
        os.makedirs(save_root, exist_ok=True)
        file_name = acq + '.nii.gz'
        FileSave(vol.squeeze(), os.path.join(save_root, file_name))
    except Exception as ex:
        print(ex)
