import os
import numpy as np
import scipy.io as sio
import h5py
from tqdm import tqdm

root = r"E:\Datasets\4Hadya\ADNI_MATs_FullVols\MPRAGEfully_IXIRotate\1DVarden30Mask"
h5file = r"E:\Datasets\4Hadya\ADNI_MATs_FullVols\MPRAGEfully_IXIRotate\1DVarden30Mask.hdf5"


n_mats = len([name for name in os.listdir(root) if os.path.isfile(os.path.join(root,name))])
h5 = h5py.File(h5file, "w")

with tqdm(total=n_mats) as pbar:
    for i in range(n_mats):
        mat = sio.loadmat(os.path.join(root,str(i)+'.mat'))
        h5ds = h5.create_group(str(i))

        for k, v in mat.items():
            if not k.startswith('__'):
                if v.dtype.type is np.str_:
                    v = v.astype('S')
                h5ds.create_dataset(k,data=v)
        pbar.update(1)

