import sys
import os
import socket
import importlib
import scipy.io as sio

## This is the name file which replaces all the different DS2MATs.

mode = 'SingleSlice' #'Vol' or 'SingleSlice' #If specified SingleSlice, then seperate MAT or seperate HDF5 groups will be created for each slice (AKA: Multi2Single). If specified as Vol then creates one MAT or one HDF5 group for each
filetype = 'MAT' #'MAT' or'H5' #Can specify H5 or MAT as mode

if filetype == 'H5' and mode == 'SingleSlice':
    from utils.TorchDS.H5Handles.MultiTorchDS2SingleH5 import TorchDS2H5 as DSHandler
elif filetype == 'H5' and mode == 'Vol':
    from utils.TorchDS.H5Handles.TorchDS2H5 import TorchDS2H5 as DSHandler
elif filetype == 'MAT' and mode == 'SingleSlice':
    from utils.TorchDS.MATHandles.MultiTorchDS2SingleMAT import TorchDS2MAT as DSHandler
elif filetype == 'MAT' and mode == 'Vol':
    from utils.TorchDS.MATHandles.TorchDS2MAT import TorchDS2MAT as DSHandler
else:
    sys.exit("Invalid filetype or mode")

mat_contents = sio.loadmat(r"/mnt/BMMR/data/Soumick/DS6NCC1701/Hendrik/Under/2DVarden05Mask.mat")
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
filename_filter = None#['Guys','HH','IOP']#None#None#'mpr-1' #mpr-1 for OASIS, for rest send None
splitXSLX = None#r'E:\Datasets\IXI\hh_split.xlsx'
split2use = None#'val'

#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
#dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
# dsClass = importlib.import_module('utils.TorchDS.ADNI').MRITorchDS #For DS following OASIS1 Standard
dsClass = importlib.import_module('utils.TorchDS.NiftiDS').MRITorchDS
#dsClass = importlib.import_module('utils.TorchDS.IXI').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.CHAOS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS

#### Define path to dataset and also to output - based on machine name
path_fully = r'/mnt/BMMR/data/Soumick/DS6NCC1701/Hendrik/Fully/val' 
path_under = r'/mnt/BMMR/data/Soumick/DS6NCC1701/Hendrik/Under/2DVarden05Mask/val'
entension_under = 'nii.gz'
output_path = r'/mnt/BMMR/data/Soumick/DS6NCC1701/Hendrik/MATs/val'
output_folder = '2DVarden05Mask' #in case of mode == 'H5', put the file name here
output_path = os.path.join(output_path, output_folder)

if filetype == 'H5' and not (output_path.endswith('.h5') or output_path.endswith('.hdf5')):
    output_path += '.hdf5'
        
obj = DSHandler(domain, IsRadial, modelType, pinMemory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, path_fully, path_under, entension_under, getROIMode, mask, filename_filter, splitXSLX, split2use)
obj.generateMATs(output_path)