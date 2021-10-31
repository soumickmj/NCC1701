import os
from pathlib import Path
from utils.HandleNifti import FileSave, FileRead3D
import cv2

root = r'E:\Datasets\OASIS-Lesion\fully'
output_root = r'E:\Datasets\OASIS-Lesion\fully_rotatedcorrect'

for filename in Path(root).glob('**/*.nii.gz'):
    filename = str(filename)
    print(filename)
    try:
        savefilename = filename.replace(root, output_root)
        vol = FileRead3D(filename).squeeze()
        for sl in range(vol.shape[2]):          
            ##For OASIS Original Orientation
            #vol[...,sl] = cv2.flip(vol[...,sl], 0) #Horizontal Flip. For Vertical the second param will be 1 - to reach same as OASIS original DS
                    
            #For OASIS Rotated Orientation (which is used in NN)
            #vol[...,sl] = cv2.flip(vol[...,sl], 1)
            #vol[...,sl] = cv2.transpose(vol[...,sl])    

            ##For shit
            vol[...,sl] = cv2.flip(vol[...,sl], 1) #Vertical Flip
            vol[...,sl] = cv2.transpose(vol[...,sl]) #Rotate Clockwise (left) - step 1
            vol[...,sl] = cv2.flip(vol[...,sl], 1) #Rotate Clockwise (left) - step 1
        os.makedirs(os.path.dirname(savefilename), exist_ok=True)
        FileSave(vol, savefilename)
    except Exception as ex:
        print(ex)