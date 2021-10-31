import h5py
import numpy as np
from matplotlib import pyplot as plt

import  utils.MRITorchDSTransforms as T
import utils.fastMRI.TorchDSTransforms as transforms

#file = r"E:\Datasets\FastMRI\raw\multicoil_train\multicoil_train\file1000001.h5"
#hf = h5py.File(file)
##volume_kspace = hf['kspace'][()]
#volume_kspace = hf['kspace'][10]
#k = transforms.to_tensor(volume_kspace)

#k = k[...,0] - k[...,1]

#import torch
##k = kspace.permute(0,3,1,2)
##k = torch.stack((k[...,0],k[...,1]), dim=1).view(k.shape[0]*2,k.shape[1],k.shape[2])
#k = torch.stack((k[...,0],k[...,1]), dim=1).view((k.shape[0]*2,) + k.shape[1:-1])

#volume_reco = hf['reconstruction_rss'][()]


from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
import os

files = glob(r'E:\Datasets\FastMRI\raw\multicoil_onlyone\multicoil_onlyone\*.h5')

headers = []
for file in files:
    hf = h5py.File(file)
    h=hf['ismrmrd_header'][()]
    headerdict = {}
    headerdict['file'] = os.path.basename(file)
    ns = {'ismrmd': 'http://www.ismrm.org/ISMRMRD'}
    root = ET.fromstring(h)
    measinfo = root.find('ismrmd:measurementInformation', ns)
    headerdict['protocol'] = measinfo.find('ismrmd:protocolName', ns).text
    pos = measinfo.find('ismrmd:patientPosition', ns).text
    acqinfo = root.find('ismrmd:acquisitionSystemInformation', ns)
    headerdict['scannermodel'] = acqinfo.find('ismrmd:systemModel', ns).text
    headerdict['tesla'] = 1.5 if headerdict['scannermodel'] == 'Aera' else 3
    headerdict['ncoil'] = acqinfo.find('ismrmd:receiverChannels', ns).text
    encoding = root.find('ismrmd:encoding', ns).find('ismrmd:encodedSpace', ns).find('ismrmd:matrixSize', ns)
    headerdict['encodingMatrix'] = (encoding.find('ismrmd:x', ns).text, encoding.find('ismrmd:y', ns).text, encoding.find('ismrmd:z', ns).text)
    recon = root.find('ismrmd:encoding', ns).find('ismrmd:reconSpace', ns).find('ismrmd:matrixSize', ns)
    headerdict['reconMatrix'] = (recon.find('ismrmd:x', ns).text, recon.find('ismrmd:y', ns).text, recon.find('ismrmd:z', ns).text)
    headerdict['nslice'] = int(root.find('ismrmd:encoding', ns).find('ismrmd:encodingLimits', ns).find('ismrmd:slice', ns).find('ismrmd:maximum', ns).text)+1
    headers.append(headerdict)

frame = pd.DataFrame(headers)
frame.to_excel(r'E:\Datasets\FastMRI\raw\multicoil_onlyone\headers.xlsx')


tensor = T.to_tensor(volume_kspace)
im_tensor = T.ifft2(tensor, True)
im_tensor2 = T.ifft2(volume_kspace, False)

print('test')