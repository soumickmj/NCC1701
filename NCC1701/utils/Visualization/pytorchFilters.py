import os
import torch
from utils.HandleNifti import FileSave

# the path for the network to load
fileToLoad  = r"D:\CloudData\OneDrive\OvGU\My Codes\Neural\Enterprise\NCC1701\NCC1701\checkpoint.pth.tar"
outputFolder = r'D:\Output\Filters\AttemptX'



### VISUALIZE FILTERS #########################################################
# load the network into the variable 'net'
checkpoint = torch.load(fileToLoad)
best_accuracy = checkpoint['best_accuracy']
epoch = checkpoint['epoch']
model = checkpoint['model']
optimizer = checkpoint['optimizer']
state_dict = checkpoint['state_dict']

## figuring out dimensions of filters, layers, etc.
for k, v in state_dict.items():
    print(k)
    print(v.shape)
    #if(k.endswith('.weight')):
    #    v_numpy = v.cpu().numpy()
    #    v_numpy = v_numpy.transpose(2,3,1,0)
    #    file_name = os.path.join(outputFolder, k + '.nii.gz')
    #    FileSave(v_numpy, file_name)