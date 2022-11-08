import SimpleITK as sitk
import numpy as np
from glob import glob
from tqdm import tqdm
import os

def sitkReader(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    return reader.Execute()

def sitkWriter(img, path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(img)

input_root = "/run/media/soumick/Enterprise/Datasets/ADNI/NewSet2022/Downloaded"
output_root = "/run/media/soumick/Enterprise/Datasets/ADNI/NewSet2022/IXIProcessed/wPad256"
pad = True

for f in tqdm(glob(f"{input_root}/**/*.nii*", recursive=True)):
    img = sitkReader(f)
    arr = sitk.GetArrayFromImage(img)
    arr = np.flip(np.transpose(arr, [2,0,1]), axis=2)
    if pad:
        arr = np.pad(arr, ((0,0),(0,0), (8,8)))
    img = sitk.GetImageFromArray(arr)
    save_path = f.replace(input_root, output_root)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitkWriter(img, save_path)