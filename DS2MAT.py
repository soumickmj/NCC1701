import os
import numpy as np
import socket
import importlib
import scipy.io as sio
from utils.TorchDS.MATHandles.TorchDS2MAT import TorchDS2MAT


######Update Param Zone#########
sliceno = 50 #Only for 2DSingleSlice
IsRadial = False
path_fully = r'D:\Desktop_migration\Medical Systems Engineering\Research Project\Juypeter\Dataset\Original\Fullysampled\Test'
path_under = r'D:\Desktop_migration\Medical Systems Engineering\Research Project\Juypeter\Dataset\Original\Undersampled\Uniform28-4step\Test'
entension_under = 'nii.gz'
output_path = r'D:\Desktop_migration\Medical Systems Engineering\Research Project\Juypeter\Dataset\MATs\Testset'
output_folder = 'Uniform28-4step'
#######Update param zone ends here###############



#### Basic parameters
neumann_name = "c503.vc-a"
domain = 'image' #In which domain, network should work? Acceptable values: | fourier | real_fourier | abs_of_fourier | real_of_fourier | abs_of_real_fourier | real_of_real_fourier | hartley | image |

sliceno = sliceno - 1 #As array starts from 0
startingSelectSlice = 65 #Only for 2DSelectMultiSlice and 3DSelectSlice
startingSelectSlice = startingSelectSlice - 1 #As array starts from 0
endingSelectSlice = 72 #Only for 2DSelectMultiSlice and 3DSelectSlice
modelType = '2DSingleSlice' #Type of Model, it can be 3D, 2DMultiSlice, 2DSingleSlice, 2D, 2DSelectMultiSlice, 3DSelectSlice
getROIMode=None #It can be 'None' if we don't need to crop out non-ROI, or 'FromUnderToFully' or 'FromFullyToUnder'
pinMemory = False

output_path = os.path.join(output_path, output_folder)


mask = np.zeros((256,256))

#### PyTorch Custom DS Class to use
#dsClass = importlib.import_module('utils.TorchDS.AlexDS2_NoMask').MRITorchDS 
dsClass = importlib.import_module('utils.TorchDS.OASIS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.CHAOS').MRITorchDS #For DS following OASIS1 Standard
#dsClass = importlib.import_module('utils.TorchDS.Alex').MRITorchDS #For Alex's 1 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex3Mask').MRITorchDS #For Alex's 3 Channel DS
#dsClass = importlib.import_module('utils.TorchDS.Alex6Mask').MRITorchDS #For Alex's 6 Channel DS
       
obj = TorchDS2MAT(domain, IsRadial, modelType, pinMemory, dsClass, sliceno, startingSelectSlice, endingSelectSlice, path_fully, path_under, entension_under, getROIMode, mask)
obj.generateMATs(output_path)