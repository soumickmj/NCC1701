####Niftis to H5
#import os
#import glob
#import h5py
#import numpy as np
#from utils.HandleNifti import FileRead3D
#path = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\T1_sag_1'
#h5file = r'D:\Datasets\Alex\T1_sag_1_for_Soumick\ds.h5'


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

import os
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.MRITorchDSTransforms import CorruptByMotionTorch, NormalizeTensor
#from model.Attention.ResNet.ResNetClassifier2D import ResNetClassifier
from model.Wavelet.BasicWaveletNet import ImageCorrectNClassifyNet

class MoCoH5DS(Dataset):
    def __init__(self, hdf5_path, rotation_limit=6.0, corruption_dir=0, trans_x_limit=0, trans_y_limit=0, normtype='minmax', transforms=None, trans4orig=None, device=torch.device('cuda:0')):
        ###Corruption Related Params
        #rotation_limit: a floating point value, defining the angle for rotation
        #corruption_dir: can be 0 or 1 or None. If None, then randomly (either 0 or 1) will be chosen for each image 
        #trans_x_limit and trans_y_limit: 0 if no translation is desired in that particular axis. otherwise any integer. ideally n//32
        ####Normalization related param - normtype: minmax or divmax 
        ####Transformation related params - augmentation to be supplied here if needed. Don't supply ToTensor() or Normalize()
        #transforms: to be applied on the corrupted slice
        #trans4orig: to be applied on the original slice
        self.h5_file = h5py.File(hdf5_path, 'r')     
        self.len = self.h5_file['subjects'].shape[-1]
        self.corruptor = CorruptByMotionTorch(rotation_limit, corruption_dir, trans_x_limit, trans_y_limit)
        self.normalizer = NormalizeTensor(normtype)
        self.transforms = transforms
        self.trans4orig = trans4orig
        self.device = device

    def __getitem__(self, index):
        slice = torch.from_numpy(self.h5_file['slices'][:,:,index]).float().to(self.device)
        subject = self.h5_file['subjects'][index]
        sliceinfo = self.h5_file['scaninfos'][index]

        corrupted, corruptions = self.corruptor(slice)

        if self.transforms is not None:
            corrupted = self.transforms(corrupted)
        if self.trans4orig is not None:
            slice = self.trans4orig(slice)

        datum = {}
        datum['input'] = self.normalizer(corrupted.unsqueeze(0))
        datum['gt'] = self.normalizer(slice.unsqueeze(0))
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
    test_name = 'WaveletImageCorrectNClassifyNet_Trainable_DO_NOBranchSplit_BothMSELoss_Run2'
    num_epochs = 100
    loader_params = {'batch_size': 4, 'shuffle': False, 'num_workers': 0} #shuffle not needed as using SubsetRandomSampler
    device = torch.device('cuda:0')
    log_freq = 10
    validation_split = 0.3
    shuffle_dataset = False #For shuffleing the split. Not ideal in our case if we want patient specific
    manul_seed = 1701

    if manul_seed is not None:
        torch.manual_seed(manul_seed)
        np.random.seed(manul_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    tb_writer = SummaryWriter(log_dir = os.path.join(log_path,test_name))

    # Create Dataset, splits and data loader
    dataset = MoCoH5DS(h5path, device=device)
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

    model = ImageCorrectNClassifyNet(n_class=1,trainable_wavelets=True, do_drop=True, branch_split=False)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    ##todo split in train, test, val


    for epoch in range(num_epochs):
        #Train
        model.train()
        runningLoss = 0.0
        runningLoss_Class = 0.0
        runningLoss_Img = 0.0
        runningLossCounter = 0.0
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            images = Variable(data['input'])
            original_images = Variable(data['gt'])
            labels = Variable(data['rotation'].to(device)).unsqueeze(1)
            optimizer.zero_grad()
            class_output, img_output = model(images)
            loss_class = loss_fn(class_output, labels)
            loss_image = loss_fn(img_output, original_images)
            loss = loss_class + loss_image
            loss.backward()
            optimizer.step()
            loss = round(loss.data.item(),4)
            train_loss += loss
            runningLoss += loss
            runningLoss_Class += round(loss_class.data.item(),4)
            runningLoss_Img += round(loss_image.data.item(),4)
            runningLossCounter += 1
            print('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss))
            #For tensorboard
            if i % log_freq == 0:
                niter = epoch*len(train_loader)+i
                tb_writer.add_scalar('Train/Loss', runningLoss/runningLossCounter, niter)
                tb_writer.add_scalar('Train/ClassLoss', runningLoss_Class/runningLossCounter, niter)
                tb_writer.add_scalar('Train/ImgLoss', runningLoss_Img/runningLossCounter, niter)
                runningLoss = 0.0
                runningLoss_Class = 0.0
                runningLoss_Img = 0.0
                runningLossCounter = 0.0
        torch.save(model.state_dict(), os.path.join(save_path, test_name+".pth.tar"))
        tb_writer.add_scalar('Train/AvgLossEpoch', train_loss/len(train_loader), epoch)
        #adjust_learning_rate(epoch)

        #Validate
        model.eval()
        runningLoss = 0.0
        runningLossCounter = 0.0
        val_loss = 0.0
        for i, data in enumerate(val_loader):
            images = Variable(data['input'])
            original_images = Variable(data['gt'])
            labels = Variable(data['rotation'].to(device)).unsqueeze(1)
            class_output, img_output = model(images)
            loss_class = loss_fn(class_output, labels)
            loss_image = loss_fn(img_output, original_images)
            loss = loss_class + loss_image
            loss = round(loss.data.item(),4)
            val_loss += loss
            runningLoss += loss
            runningLoss_Class += round(loss_class.data.item(),4)
            runningLoss_Img += round(loss_image.data.item(),4)
            runningLossCounter += 1
            print('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))
            #For tensorboard
            if i % log_freq == 0:
                niter = epoch*len(val_loader)+i
                tb_writer.add_scalar('Val/Loss', runningLoss/runningLossCounter, niter)
                tb_writer.add_scalar('Val/ClassLoss', runningLoss_Class/runningLossCounter, niter)
                tb_writer.add_scalar('Val/ImgLoss', runningLoss_Img/runningLossCounter, niter)
                runningLoss = 0.0
                runningLoss_Class = 0.0
                runningLoss_Img = 0.0
                runningLossCounter = 0.0
        tb_writer.add_scalar('Val/AvgLossEpoch', val_loss/len(val_loader), epoch)