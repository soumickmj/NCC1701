import pandas as pd
from typing import Literal, Optional, Sequence, Union, Callable
from glob import glob
import numpy as np
import torch
import torchio as tio
import os
import sys


def createTIOSubDS(
    root_gt: str,
    root_input: Optional[str] = None,
    filename_filter: Optional[Union[str, Sequence[str]]] = None,
    split_csv: str = "",
    split: str = "",
    data_mode: Literal['NIFTI', 'DICOM'] = "NIFTI",
    isKSpace: bool = False,
    isGTNonImg: bool = False,
    init_transforms: Optional[Callable] = None,
    aug_transforms: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
) -> tio.SubjectsDataset:
    if data_mode == "NIFTI":
        files = glob(root_gt+"/**/*.nii", recursive=True) + glob(root_gt+"/**/*.nii.gz", recursive=True) +\
            glob(root_gt+"/**/*.img", recursive=True) + \
            glob(root_gt+"/**/*.img.gz", recursive=True)
    else:
        # TODO: DICOM read
        sys.exit("DICOM read not implemented inside createTIOSubDS")

    if bool(filename_filter):
        if type(filename_filter) is str:
            filename_filter = [filename_filter]
        files = [f for f in files if any(
            filt in f for filt in filename_filter)]

    if bool(split_csv):
        df = pd.read_csv(split_csv)[split]      
        df.dropna(inplace=True)  
        df = list(df)
        files = [f for f in files if os.path.basename(f).split(".")[0] in df] #TODO: remember, this will give an error if the filename also has dots in the name, apart from the extension (for example abc.def.nii.gz. this will return abc only)

    subjects = []
    for file in files:
        if bool(root_input):
            file_in = file.replace(root_gt, root_input)
            subjects.append(tio.Subject(
                inp=tio.ScalarImage(file_in),
                gt=tio.LabelMap(file) if isGTNonImg else tio.ScalarImage(file),
                filename=os.path.basename(file),
                processed=False
            ))
        else:
            subjects.append(tio.Subject(
                gt=tio.ScalarImage(file),
                filename=os.path.basename(file),
                processed=False
            ))
            

    if isKSpace:
        # TODO: Image to kSpace transform
        sys.exit("Image to kSpace transform not implemented inside createTIOSubDS")

    if init_transforms is not None:
        aug_transforms = init_transforms if aug_transforms is None else (
            init_transforms+aug_transforms)

    if aug_transforms is not None:
        transforms = aug_transforms if transforms is None else (
            aug_transforms+transforms)

    if transforms is not None:
        transforms = tio.Compose(transforms)

    subjects_dataset = tio.SubjectsDataset(subjects, transform=transforms)
    return subjects_dataset

def create_patchQs(train_subs, val_subs, patch_size, patch_qlen, patch_per_vol, inference_strides): 
    train_queue = None
    val_queue = None
    grid_samplers = [] 
    
    if train_subs is not None:
        sampler = tio.data.UniformSampler(patch_size)
        train_queue = tio.Queue(
                subjects_dataset=train_subs,
                max_length=patch_qlen,
                samples_per_volume=patch_per_vol,
                sampler=sampler,
                num_workers=0,
                start_background=True
            )

    if val_subs is not None:
        stride_length, stride_width, stride_depth = inference_strides.split(',')
        overlap = np.subtract(patch_size, (int(stride_length), int(stride_width), int(stride_depth)))           
        for i in range(len(val_subs)):
            grid_sampler = tio.inference.GridSampler(val_subs[i], patch_size, overlap)
            grid_samplers.append(grid_sampler)
        val_queue = torch.utils.data.ConcatDataset(grid_samplers)

    return train_queue, val_queue, grid_samplers