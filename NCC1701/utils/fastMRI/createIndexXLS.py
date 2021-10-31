import os
import h5py
import glob
import pandas as pd
from tqdm import tqdm
import ismrmrd.xsd
from utils.RAW.ISMRMRDReader import ISMRMRDReader

root = r'E:\Datasets\FastMRI\raw\multicoil_train\multicoil_train'
XLSPath = r'E:\Datasets\FastMRI\raw\multicoil_train\multicoil_train.xlsx'

files = glob.glob(os.path.join(root, '*.h5'))
headers = []
with tqdm(total=len(files)) as pbar:
    for file in files:
        data = h5py.File(file, 'r') 
        basicinfo = dict(data.attrs)
        nSlice, nCoil, hKSP, wKSP = data['kspace'].shape
        try: #not present in the test set
            _, hIMG, wIMG = data['reconstruction_rss'].shape
        except:
            hIMG = -1
            wIMG = -1
        headerXML = data['ismrmrd_header'][()]
        headerDict = ISMRMRDReader.readHeaderFromXMLExtended(headerXML)
        headerDict = {f'zISMRM_{k}': v for k, v in headerDict.items()}
        basicinfo.update({'file': os.path.basename(file), 'nSlice':nSlice, 'nCoil':nCoil, 'hKSP':hKSP, 'wKSP':wKSP, 'hIMG':hIMG, 'wIMG':wIMG})
        headerDict.update(basicinfo)
        headers.append(headerDict)
        pbar.update(1)

df = pd.DataFrame.from_dict(headers)
df.to_excel(XLSPath)