import os
import random
import sys
from glob import glob
from typing import Callable, Literal, Optional, Sequence, Union
from tqdm import tqdm

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
import torchvision
from torch.utils.data import Dataset
from Engineering.transforms.transforms import IntensityNorm

def __read_vol(vol_path, data_mode):
    if data_mode.lower() == "nifti":
        return nib.load(vol_path).get_fdata()
    elif data_mode.lower() == "npy":
        return np.load(vol_path)

def __count_volslice(vol_path, mid_n=-1, mid_per=-1, random_n=-1):
    n_slices = nib.load(vol_path).shape[-1]
    slices = list(range(n_slices))
    if mid_n == -1 and abs(mid_per) != 1:
        mid_n = round(n_slices * mid_per)
    if mid_n != -1:
        strt_idx = (len(slices) // 2) - (mid_n // 2)
        end_idx = (len(slices) // 2) + (mid_n // 2)
        slices = slices[strt_idx: end_idx + 1]
    elif random_n != -1:
        slices = random.choices(slices, k=random_n)
    return {"path": [vol_path] * len(slices), "sliceID": slices}


class MRITorchDS(Dataset):
    def __init__(self, df, is3D=True, transform=None, expand_ch=True) -> None:
        self.df = df
        self.is3D = is3D
        self.transform = transform
        self.expand_ch = expand_ch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        datum = self.df.iloc[idx]
        gt = (
            np.array(nib.load(datum['gtpath']).get_fdata())
            if self.is3D
            else np.array(nib.load(datum['gtpath']).dataobj[..., datum['sliceID']])
        )

        sample = {
            "gt": {
                "data": gt.astype(np.float32),
                "path": datum['gtpath']
            },
            "filename": datum["filename"],
            "sliceID": datum["sliceID"]
        }
        if "gtmax" in datum:
            sample["gt"]["volmax"] = datum["gtmax"]
            sample["gt"]["volmin"] = datum["gtmin"]
        if "inpath" in datum:
            inp = (
                np.array(nib.load(datum['inpath']).get_fdata())
                if self.is3D
                else np.array(
                    nib.load(datum['inpath']).dataobj[..., datum['sliceID']]
                )
            )

            sample["inp"] = {
                "data": inp.astype(np.float32),
                "path": datum['inpath']
            }
            if "inmax" in datum:
                sample["inp"]["volmax"] = datum["inmax"]
                sample["inp"]["volmin"] = datum["inmin"]
        if bool(self.transform):
            sample = self.transform(sample)
        if self.expand_ch:
            sample['gt']['data'] = np.expand_dims(sample['gt']['data'], 0)
            if 'inp' in sample:
                sample['inp']['data'] = np.expand_dims(
                    sample['inp']['data'], 0)
        return sample


def createFileDS(
    root_gt: Union[str, Sequence[str]],
    root_input: Optional[Union[str, Sequence[str]]] = None,
    filename_filter: Optional[Union[str, Sequence[str]]] = None,
    processed_csv: str = "",
    split_csv: str = "",
    split: str = "",
    data_mode: Literal['NIFTI', 'NPY', 'DICOM'] = "NIFTI",
    isKSpace: bool = False,
    isGTNonImg: bool = False,
    init_transforms: Optional[Callable] = None,
    aug_transforms: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
    is3D: bool = False,
    mid_n: int = -1,  # Only if is3D=False
    mid_per: float = -1,  # Only if is3D=False
    random_n: int = -1,  # Only if is3D=False
) -> MRITorchDS:

    if not bool(processed_csv) or not os.path.isfile(processed_csv):
        if type(root_gt) is not list:
            root_gt = [root_gt]

        if bool(root_input) and type(root_input) is not list:
            root_input = [root_input]

        if not is3D:
            normtype = next((x.type for x in init_transforms if isinstance(x, IntensityNorm)), "")

        files = []
        for gt in root_gt:
            if data_mode.lower() == "nifti":
                files += glob(gt+"/**/*.nii", recursive=True) + glob(gt+"/**/*.nii.gz", recursive=True) +\
                    glob(gt+"/**/*.img", recursive=True) + \
                    glob(gt+"/**/*.img.gz", recursive=True)
            elif data_mode.lower() == "npy":
                files += glob(gt+"/**/*.npy", recursive=True)
            else:
                # TODO: DICOM read
                sys.exit("DICOM read not implemented inside createFileDS")

        if bool(filename_filter):
            if type(filename_filter) is str:
                filename_filter = [filename_filter]
            files = [f for f in files if any(
                filt in f for filt in filename_filter)]

        if bool(split_csv):
            df = pd.read_csv(split_csv)[split]
            df.dropna(inplace=True)
            df = list(df)
            files = [f for f in files if any(d in f for d in df)]

        data_dfs = []
        filenames = []
        print("Preparing dataset .....")
        for file in tqdm(files):
            if not os.path.isfile(file):
                continue
            filenames.append(os.path.basename(file))
            if not is3D:
                datum_dict = __count_volslice(
                    file, mid_n=mid_n, mid_per=mid_per, random_n=random_n)
                datum_dict["gtpath"] = datum_dict.pop("path")
                if "vol" in normtype:
                    v = nib.load(file).get_fdata()
                    datum_dict["gtmin"] = [v.min()] * len(datum_dict["sliceID"])
                    datum_dict["gtmax"] = [v.max()] * len(datum_dict["sliceID"])
            else:
                datum_dict = {
                    "gtpath": [file],
                    "sliceID": [-1]
                }
            datum_dict["filename"] = [filenames[-1]] * len(datum_dict["sliceID"])
            if bool(root_input):
                gt_id = [i for i, g in enumerate(root_gt) if g in file][0]
                file_in = file.replace(root_gt[gt_id], root_input[gt_id])
                if not os.path.isfile(file_in):
                    if data_mode != "NIFTI" or ".gz" in file_in:
                        continue

                    file_in += ".gz"
                    if not os.path.isfile(file_in):
                        continue
                datum_dict["inpath"] = [file_in] * len(datum_dict["sliceID"])
                if "vol" in normtype:
                    v = nib.load(file_in).get_fdata()
                    datum_dict["inmin"] = [v.min()] * len(datum_dict["sliceID"])
                    datum_dict["inmax"] = [v.max()] * len(datum_dict["sliceID"])
            data_dfs.append(pd.DataFrame.from_dict(datum_dict))
        data_df = pd.concat(data_dfs).reset_index(drop=True)

        if bool(processed_csv):
            data_df.to_csv(processed_csv)
    else:
        data_df = pd.read_csv(processed_csv)
        filenames = data_df.filename.unique().tolist()

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
        transforms = torchvision.transforms.Compose(transforms)

    dataset = MRITorchDS(data_df, is3D=is3D, transform=transforms)
    return dataset, filenames

