import os
import socket
import numpy as np
import importlib
import scipy.io as sio
from utils.TorchDS.MATHandles.MultiTorchDS2SingleMAT import TorchDS2MAT


######Update Param Zone#########
startingSelectSlice = 1 #for whole vol start from 1
endingSelectSlice = 166 #for whole vol end at 128,  166
IsRadial = False
path_fully = r'Z:\ADNI\NIFTIs\MP-RAGEfully_IXIRotate'
path_under = r'X:\user\hyassin\ADNI\Undersampled\MP-RAGEunder_IXIRotate\Radial60GA'  
entension_under = 'nii.gz'
output_path = r'D:\Hadya\Research Project\Juypeter\ADNI\MATs\MP-RAGE_IXIRotate'
output_folder = 'Radial60GA_Sl75-90'                                                 

# Don't forget to change PyTorch Custom DS Class to use down in the code for different datasets
# such as; OASIS, ADNI

#######Update param zone ends here###############



#### Basic parameters
neumann_name = "c503.vc-a"
domain = 'image' #In which domain, network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image |
sliceno = 50 #Only for 2DSingleSlice
sliceno = sliceno - 1 #As array starts from 0
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DSelectMultiSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
getROIMode=None #It can be 'None' if we don't need to crop out non-ROI, or 'FromUnderToFully' or 'FromFullyToUnder'
pinMemory = False
output_path = os.path.join(output_path, output_folder)

#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
dsClass = importlib.import_module('utils.TorchDS.ADNI').MRITorchDS #For DS following ADNI
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.CHAOS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS

mask = np.zeros((256,256))
        
obj = TorchDS2MAT(domain, IsRadial, modelType, pinMemory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, path_fully, path_under, entension_under, getROIMode, mask)
obj.generateMATs(output_path)