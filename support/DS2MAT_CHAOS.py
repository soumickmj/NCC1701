import os
import socket
import importlib
import scipy.io as sio
from utils.TorchDS.MATHandles.TorchDS2MAT import TorchDS2MAT

mat_contents = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVardenMask.mat")
mask = mat_contents['mask']

#### Basic parameters
neumann_name = "c503.vc-a"
domain = 'fourier_magphase' #In which domain, network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image |
sliceno = 1 #Only for 2DSingleSlice
sliceno = sliceno - 1 #As array starts from 0
startingSelectSlice = 65 #Only for 2DSelectMultiSlice and 3DSelectSlice
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
endingSelectSlice = 72 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DSelectMultiSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
getROIMode=None #It can be 'None' if we don't need to crop out non-ROI, or 'FromUnderToFully' or 'FromFullyToUnder'
pinMemory = False
IsRadial = False

#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.CHAOS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS

#### Define path to dataset and also to output - based on machine name
pc_name = socket.gethostname()
if(pc_name == 'BMMR-Soumick'):
    ##For BMMR-Soumick - For Undersampled Images
    path_fully = r'D:\Datasets\CHAOS_dataset\CT_NII'
    path_under = r'D:\Datasets\OASIS1\Undersampled-Registered\TestSet\varden30SameMask'
    entension_under = 'nii.gz'
    output_path = r'D:\Datasets\MATs\CHAOS_CT'
    output_folder = 'data'
    output_path = os.path.join(output_path, output_folder)
elif(pc_name == 'powerzon04'):
    print('not defined')
elif(pc_name == 'quadro'):
    print('not defined')
elif(pc_name == neumann_name):
    print('not defined')
else:
    sys.exit('PC Name Recognized. So, path definations not initialized')
        
obj = TorchDS2MAT(domain, IsRadial, modelType, pinMemory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, path_fully, path_under, entension_under, getROIMode, mask)
obj.generateMATs(output_path)