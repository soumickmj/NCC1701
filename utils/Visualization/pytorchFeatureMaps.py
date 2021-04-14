import os
import json
import scipy.io as sio
import numpy as np
import torch
from utils.HandleNifti import FileSave

# the path for the network to load
fileToLoad  = r'D:\Output\Attempt19-WithValidate-OASISVarden30SameMask-ResNet2Dv2SSIMLoss-SingleSlice50\checkpoints\model_best.pth.tar'
outputFolder = r'D:\Output\FeatureMaps\Attempt19\testIMGDomeMe_us'

#input = torch.randn(1, 1, 256, 256)
#OR
#input_mat_path = r'D:\Datasets\MATs\OASIS-Subset1-varden30SameMask-Slice50\ds1-image-only1test\0.mat'
input_mat_path = r'C:\Users\soumi\OneDrive\Pictures\testIMG_us.mat'
mat = sio.loadmat(input_mat_path)
input = mat['under']
input = np.expand_dims(input, 0)
input = torch.from_numpy(input).float()
#OR
#input = torch.zeros((1,1,256,256))
#OR
#input = torch.ones((1,1,256,256))
#OR


### VISUALIZE FILTERS #########################################################
# load the network into the variable 'net'
checkpoint = torch.load(fileToLoad)
best_accuracy = checkpoint['best_accuracy']
epoch = checkpoint['epoch']
model = checkpoint['model']
optimizer = checkpoint['optimizer']
state_dict = checkpoint['state_dict']

index = 0
activation = {}
activation_names = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        activation_names[name] = str(model)
    return hook
for i in range(len(model.model)):
    if 'Conv' in str(model.model[i]):
        model.model[i].register_forward_hook(get_activation(str(index)))
        index += 1

input = input.cuda()
output = model.model(input)

for k, v in activation.items():
    v_numpy = v.cpu().numpy()
    v_numpy = v_numpy.transpose(2,3,1,0)
    file_name = os.path.join(outputFolder, k + '.nii.gz')
    FileSave(v_numpy, file_name)

json = json.dumps(activation_names, indent=4)
file_name = os.path.join(outputFolder, 'activation_names.json')
f = open(file_name,"w")
f.write(json)
f.close()

input_numpy = input.cpu().numpy()
input_numpy = input_numpy.transpose(2,3,1,0)
file_name = os.path.join(outputFolder, 'input.nii.gz')
FileSave(input_numpy, file_name)

output_numpy = output.detach().cpu().numpy()
output_numpy = output_numpy.transpose(2,3,1,0)
file_name = os.path.join(outputFolder, 'output.nii.gz')
FileSave(output_numpy, file_name)

print('done')