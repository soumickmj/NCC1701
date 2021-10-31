####Niftis to H5
#import os
#import glob
#import h5py
#import numpy as np
#from utils.HandleNifti import FileRead3D
#path = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\MINIXI'
#h5file = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\minixi.h5'


#niftis = glob.glob(os.path.join(path, '*.nii.gz'))
#slice_select_start = 38
#slice_select_end = 138
#h5 = h5py.File(h5file, "w")

#subjects = []
#scaninfos = []
#vols = None
#for nifti in niftis:
#    vol = FileRead3D(nifti).squeeze()[:,:,slice_select_start:slice_select_end]
#    temp = os.path.basename(nifti).split('_')
#    subject = temp[0]
#    scandetails = temp[1].split('.')[0]
#    if subject in subjects:
#        continue
#    if vols is None:
#        vols = vol
#    else:
#        vols = np.concatenate([vols,vol],axis=-1)
#    subjects += [subject] * (slice_select_end-slice_select_start)
#    scaninfos += [scandetails+'-sl'+"{0:0=3d}".format(sl) for sl in range(slice_select_start,slice_select_end)]

#h5.create_dataset('slices',data=vols)
#h5.create_dataset('subjects',data=np.string_(subjects))
#h5.create_dataset('scaninfos',data=np.string_(scaninfos))

#print('done')

import random
import os
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.MRITorchDSTransforms import CorruptByMotionTorch, CorruptByMotionNP, NormalizeTensor
#from model.Attention.ResNet.ResNetClassifier2D import ResNetClassifier
from model.Wavelet.BasicWaveletNet import ImageCorrectNClassifyNet

manul_seed = 1701


if manul_seed is not None:
    torch.manual_seed(manul_seed)
    np.random.seed(manul_seed)
    random.seed(manul_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
else:
    cudnn.benchmark = True

#class MoCoH5DS(Dataset):
#    def __init__(self, hdf5_path, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, rotatenorm=True, normtype='minmax', transforms=None, trans4orig=None, device=torch.device('cuda:0')):
#        ###Corruption Related Params
#        #rotation_limit: a floating point value, defining the angle for rotation
#        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
#        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
#        ####Normalization related param - normtype: minmax or divmax 
#        ####Transformation related params - augmentation to be supplied here if needed. Don't supply ToTensor() or Normalize()
#        #transforms: to be applied on the corrupted slice
#        #trans4orig: to be applied on the original slice
#        self.h5_file = h5py.File(hdf5_path, 'r')     
#        self.len = self.h5_file['subjects'].shape[-1]
#        self.corruptor = CorruptByMotionTorch(rotation_limit, corruption_dir, trans_x_limit, trans_y_limit, rotatenorm)
#        self.normalizer = NormalizeTensor(normtype)
#        self.transforms = transforms
#        self.trans4orig = trans4orig
#        self.device = device

#    def __getitem__(self, index):
#        slice = torch.from_numpy(self.h5_file['slices'][:,:,index]).float().to(self.device)
#        subject = self.h5_file['subjects'][index]
#        sliceinfo = self.h5_file['scaninfos'][index]

#        corrupted, corruptions = self.corruptor(slice)

#        if self.transforms is not None:
#            corrupted = self.transforms(corrupted)
#        if self.trans4orig is not None:
#            slice = self.trans4orig(slice)

#        datum = {}
#        datum['input'] = self.normalizer(corrupted.unsqueeze(0))
#        datum['gt'] = self.normalizer(slice.unsqueeze(0))
#        datum['subject'] = subject.decode('UTF-8')
#        datum['sliceinfo'] = sliceinfo.decode('UTF-8')
#        #corruptions = {k: torch.from_numpy(np.array([v])).float() for k, v in corruptions.items()}
#        corruptions = {k: torch.tensor(v).float() for k, v in corruptions.items()}
#        datum.update(corruptions)

#        return datum

#    def __len__(self):
#        return self.len

class MoCoH5DS(Dataset):
    def __init__(self, hdf5_path, rotation_limit=10.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, rotatenorm=True, normtype='minmax', transforms=None, trans4orig=None, device=torch.device('cuda:0')):
        ###Corruption Related Params
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        ####Normalization related param - normtype: minmax or divmax 
        ####Transformation related params - augmentation to be supplied here if needed. Don't supply ToTensor() or Normalize()
        #transforms: to be applied on the corrupted slice
        #trans4orig: to be applied on the original slice
        h5_file = h5py.File(hdf5_path, 'r')  
        x=[]
        for s in h5_file['subjects']:
            x.append(s)
        print(set(x))
        self.len = h5_file['subjects'].shape[-1]
        self.hdf5_path = hdf5_path
        #self.corruptor = CorruptByMotionTorch(rotation_limit, corruption_dir, trans_x_limit, trans_y_limit, rotatenorm)
        self.corruptor = CorruptByMotionNP(rotation_limit, corruption_dir, trans_x_limit, trans_y_limit, rotatenorm)
        self.normalizer = NormalizeTensor(normtype)
        self.transforms = transforms
        self.trans4orig = trans4orig
        self.device = device

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path,'r') as h5_file:
            #slice = torch.from_numpy(h5_file['slices'][:,:,index]).float().to(self.device)
            slice = h5_file['slices'][:,:,index]
            subject = h5_file['subjects'][index]
            sliceinfo = h5_file['scaninfos'][index]

        corrupted, corruptions = self.corruptor(slice)

        if self.transforms is not None:
            corrupted = self.transforms(corrupted)
        if self.trans4orig is not None:
            slice = self.trans4orig(slice)

        datum = {}
        #datum['input'] = self.normalizer(corrupted.unsqueeze(0))#.cpu()
        #datum['gt'] = self.normalizer(slice.unsqueeze(0))#.cpu()
        datum['input'] = self.normalizer(np.expand_dims(corrupted,0))#.cpu()
        datum['gt'] = self.normalizer(np.expand_dims(slice,0))#.cpu()
        datum['subject'] = subject.decode('UTF-8')
        datum['sliceinfo'] = sliceinfo.decode('UTF-8')
        #corruptions = {k: torch.from_numpy(np.array([v])).float() for k, v in corruptions.items()}
        corruptions = {k: torch.tensor(v).float() for k, v in corruptions.items()}
        datum.update(corruptions)

        return datum

    def __len__(self):
        return self.len

if __name__ == '__main__':
    h5path = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\ds.h5'
    log_path = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\tb'
    save_path = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\save'
    test_name = 'ResNeXt101_PreT_BS24'
    checkpoint = None#'r"D:\Datasets\Alex\T1_sag_1_for_Soumick\save\ResNeXt101_PreT_BS24.pth.tar" #Make it None if you don't want to load
    runOnlyTest = False
    h5path4test = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\minixi.h5'
    testOutFile = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\test.mat'
    num_epochs = 50
    loader_params = {'batch_size': 24, 'shuffle': False, 'num_workers': 0} #shuffle not needed as using SubsetRandomSampler
    device = torch.device('cuda:0')
    device4reader = torch.device('cuda:1') #can use same
    log_freq = 10
    validation_split = 0.3
    shuffle_dataset = False #For shuffleing the split. Not ideal in our case if we want patient specific
    

    tb_writer = SummaryWriter(log_dir = os.path.join(log_path,test_name))

    

    #model = ImageCorrectNClassifyNet(n_class=1,trainable_wavelets=True, do_drop=True, branch_split=False, relu_at_end=False)
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
    model.fc = nn.Linear(model.fc.in_features, out_features=1)
    model.to(device)
    model.share_memory()
    #model = nn.DataParallel(model)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    ##todo split in train, test, val

    if runOnlyTest:
        dataset = MoCoH5DS(h5path4test, device=device4reader, rotatenorm=False)
        test_loader = DataLoader(dataset, **loader_params)
        #Validate
        model.eval()
        outputClasses = []
        outputImgs = []
        inputImgs = []
        originalImgs = []
        inputLabels = []
        for i, data in enumerate(test_loader):
            print(i)
            images = Variable(data['input']).cuda().float()
            original_images = Variable(data['gt'])
            labels = Variable(data['rotation'].to(device)).unsqueeze(1)
            class_output = model(images)
            outputClasses.append(class_output.cpu().detach().numpy())
            inputImgs.append(images.cpu().detach().numpy())
            originalImgs.append(original_images.cpu().detach().numpy())
            inputLabels.append(labels.cpu().detach().numpy())
            break
        import scipy.io as sio
        sio.savemat(testOutFile, {'OutLabels':outputClasses, 'Labels':inputLabels, 'InImgs': inputImgs, 'OriginalImgs': originalImgs})
    else:
        # Create Dataset, splits and data loader
        dataset = MoCoH5DS(h5path, device=device4reader, rotatenorm=False)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        #split = int(np.floor(validation_split * dataset_size))
        #if shuffle_dataset :
        #    np.random.shuffle(indices)
        split = 601 #6 subjects
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, sampler=train_sampler, **loader_params)
        val_loader = DataLoader(dataset, sampler=valid_sampler, **loader_params)

        for i, data in enumerate(train_loader):
            print(data['subject'])
        #for epoch in range(num_epochs):
        #    #Train
        #    model.train()
        #    runningLoss = 0.0
        #    runningLossCounter = 0.0
        #    train_loss = 0.0
        #    for i, data in enumerate(train_loader):
        #        images = Variable(data['input']).to(device).float()
        #        labels = Variable(data['rotation'].to(device)).unsqueeze(1)
        #        optimizer.zero_grad()
        #        class_output = model(images)
        #        loss = loss_fn(class_output, labels)
        #        loss.backward()
        #        optimizer.step()
        #        loss = round(loss.data.item(),4)
        #        train_loss += loss
        #        runningLoss += loss
        #        runningLossCounter += 1
        #        print('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss))
        #        #For tensorboard
        #        if i % log_freq == 0:
        #            niter = epoch*len(train_loader)+i
        #            tb_writer.add_scalar('Train/Loss', runningLoss/runningLossCounter, niter)
        #            runningLoss = 0.0
        #            runningLossCounter = 0.0
        #    torch.save(model.state_dict(), os.path.join(save_path, test_name+".pth.tar"))
        #    tb_writer.add_scalar('Train/AvgLossEpoch', train_loss/len(train_loader), epoch)
        #    #adjust_learning_rate(epoch)

        #    #Validate
        #    model.eval()
        #    runningLoss = 0.0
        #    runningLossCounter = 0.0
        #    val_loss = 0.0
        #    with torch.no_grad():
        #        for i, data in enumerate(val_loader):
        #            images = Variable(data['input']).to(device).float()
        #            labels = Variable(data['rotation'].to(device)).unsqueeze(1)
        #            class_output = model(images)
        #            loss = loss_fn(class_output, labels)
        #            loss = round(loss.data.item(),4)
        #            val_loss += loss
        #            runningLoss += loss
        #            runningLossCounter += 1
        #            print('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))
        #            #For tensorboard
        #            if i % log_freq == 0:
        #                niter = epoch*len(val_loader)+i
        #                tb_writer.add_scalar('Val/Loss', runningLoss/runningLossCounter, niter)
        #                runningLoss = 0.0
        #                runningLossCounter = 0.0
        #    tb_writer.add_scalar('Val/AvgLossEpoch', val_loss/len(val_loader), epoch)