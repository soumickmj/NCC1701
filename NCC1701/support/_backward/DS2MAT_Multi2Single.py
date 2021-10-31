import os
import socket
import importlib
import scipy.io as sio


mode = 'H5' #'MAT' #Can specify H5 or MAT as mode
if mode == 'H5':
    from utils.TorchDS.H5Handles.MultiTorchDS2SingleH5 import TorchDS2H5 as DSHandler
else:
    from utils.TorchDS.MATHandles.MultiTorchDS2SingleMAT import TorchDS2MAT as DSHandler

mat_contents = sio.loadmat(r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\1DVarden30Mask.mat")
mask = mat_contents['mask']

#### Basic parameters
neumann_name = "c503.vc-a"
domain = 'image' #In which domain, network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image |
sliceno = 50 #Only for 2DSingleSlice
sliceno = sliceno - 1 #As array starts from 0
startingSelectSlice = 60 #Only for 2DSelectMultiSlice and 3DSelectSlice
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
endingSelectSlice = 90 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DMultiSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
getROIMode=None #It can be 'None' if we don't need to crop out non-ROI, or 'FromUnderToFully' or 'FromFullyToUnder'
pinMemory = False
IsRadial = False
filename_filter = ['Guys','HH','IOP']#None#None#'mpr-1' #mpr-1 for OASIS, for rest send None
splitXSLX = r'E:\Datasets\IXI\hh_split.xlsx'
split2use = 'val'

#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.ADNI').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.IXI').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.CHAOS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS

#### Define path to dataset and also to output - based on machine name
pc_name = socket.gethostname()
if(pc_name == 'BMMR-Soumick'):
    ##For BMMR-Soumick - For Undersampled Images
    path_fully = r'E:\Datasets\ADNI\NIFTIs\MPRAGEfully_IXIRotate'
    path_under = r'E:\Datasets\ADNI\NIFTIs\MPRAGEunder_IXIRotate\1DVarden30Mask'
    entension_under = 'nii.gz'
    output_path = r'E:\Datasets\4Hadya\ADNI_MATs_FullVols\MPRAGEfully_IXIRotate\1DVarden05Mask'
    output_folder = split2use #in case of mode == 'H5', put the file name here. split2use will automatically take a name as per the split
    output_path = os.path.join(output_path, output_folder)
elif(pc_name == 'powerzon04'):
    print('not defined')
elif(pc_name == 'quadro'):
    print('not defined')
elif(pc_name == neumann_name):
    print('not defined')
else:
    sys.exit('PC Name Recognized. So, path definations not initialized')
        
if mode == 'H5' and not (output_path.endswith('.h5') or output_path.endswith('.hdf5')):
    output_path += '.h5'
        
obj = DSHandler(domain, IsRadial, modelType, pinMemory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, path_fully, path_under, entension_under, getROIMode, mask, filename_filter, splitXSLX, split2use)
obj.generateMATs(output_path)