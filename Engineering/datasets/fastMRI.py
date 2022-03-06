import os
from pathlib import Path
import random
import sys
from glob import glob
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union
import ast

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
import torchvision
from torch.utils.data import Dataset
import fastmri
from fastmri.data import SliceDataset
from fastmri.data import transforms as fastMRItransforms
from Engineering.math.complex import complex_modeconverter
from Engineering.math.freq_trans import fftNc_pyt, ifftNc_pyt
from Engineering.math.misc import root_sum_of_squares_pyt
from Engineering.transforms.transforms import cropcentreIfNeeded
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.subsample import MaskFunc


class fastMRIDataPrep:
    """
    Data Transformer.
    A combined abdaptation of UnetDataTransfrom and VarNetDataTransform
    from, https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/transforms.py
    """

    def __init__(
        self,
        which_challenge: str,
        is_test: bool = False,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        centre_crop: bool = True,
        prenorm: bool = True
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            is_test: Set if its test set of fastMRI
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(
                "Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.is_test = is_test
        self.use_seed = use_seed
        self.centre_crop = centre_crop
        self.prenorm = prenorm

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace = torch.from_numpy(kspace)
        image = ifftNc_pyt(kspace)
        if self.prenorm:
            image /= torch.abs(image).max()
            if not self.centre_crop:  # As inside centre_crop, fft will be applied anyway
                kspace = fftNc_pyt(image)

        if self.centre_crop:
            if target is not None:
                crop_size = (target.shape[-2], target.shape[-1])
            else:
                crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

            image = fastMRItransforms.center_crop(data=image, shape=crop_size)
            kspace = fftNc_pyt(image)
        else:
            crop_size = None

        # check for max value and padding info
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        acq_start = attrs["padding_left"] if "padding_left" in attrs.keys() else -1
        acq_end = attrs["padding_right"] if "padding_right" in attrs.keys() else -1
        if acq_start==-1 or acq_end==-1:
            padding = None
        else:
            padding = (acq_start, acq_end)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            under_kspace, mask, num_low_frequencies = fastMRItransforms.apply_mask(
                kspace, self.mask_func, seed=seed,  padding=padding 
            )
            attrs['num_low_frequencies'] = num_low_frequencies
        else:
            if self.is_test:
                under_kspace = kspace
            else:
                under_kspace = None
                mask = None

        # inverse Fourier transform to get zero filled solution
        under_image = ifftNc_pyt(
            under_kspace) if under_kspace is not None else None

        return {
            "underK": under_kspace,
            "fullyK": kspace,
            "underI": under_image,
            "fullyI": image,
            "mask": mask,
            "attrs": attrs,
            "fname": fname,
            "slice_num": slice_num
        }


def coilCombiner(image, kspace, do_image, do_kspace):
    if do_image and do_kspace:
        image = root_sum_of_squares_pyt(image)
        kspace = fftNc_pyt(image)
    elif do_image and not do_kspace:
        image = root_sum_of_squares_pyt(image)
    elif not do_image and do_kspace:
        _tmp = root_sum_of_squares_pyt(image)
        kspace = fftNc_pyt(_tmp)
    return image, kspace


class fastMRIDS(SliceDataset):
    def __init__(self, root: Union[str, Path, os.PathLike],
                 challenge: str,
                 dataprep: Optional[Callable] = None,
                 use_dataset_cache: bool = False,
                 sample_rate: Optional[float] = None,
                 volume_sample_rate: Optional[float] = None,
                 dataset_cache_file: Union[str, Path,
                                           os.PathLike] = "dataset_cache.pkl",
                 num_cols: Optional[Tuple[int]] = None,
                 transform: Optional[Callable] = None,
                 expand_ch: bool = True,
                 combine_coils: Optional[Dict] = None,
                 complex_image_modes: Optional[Dict] = None):
        super().__init__(root=root, challenge=challenge,
                         transform=dataprep, use_dataset_cache=use_dataset_cache,
                         sample_rate=sample_rate, volume_sample_rate=volume_sample_rate,
                         dataset_cache_file=dataset_cache_file, num_cols=num_cols)
        self.custom_transform = transform
        self.expand_ch = expand_ch
        self.combine_coils = combine_coils
        self.complex_image_modes = complex_image_modes

    def __getitem__(self, idx):
        datum = super(fastMRIDS, self).__getitem__(idx)
        sample = {
            "gt": {
                "data": datum['fullyI'],
                "ksp": datum['fullyK'],
                "path": datum["fname"]
            },
            "filename": datum["fname"],
            "sliceID": datum["slice_num"],
            "metadict": {
                "fastMRIAttrs": datum["attrs"],
            }
        }

        if datum["mask"] is not None:
            sample["metadict"]["mask"] = datum["mask"]

        if "underK" in datum and "underI" in datum and datum["underK"] is not None and datum["underI"] is not None:
            sample["inp"] = {
                "data": datum['underI'],
                "ksp": datum['underK'],
                "path": datum["fname"]
            }

        del datum
        
        if bool(self.combine_coils) and type(self.combine_coils) is dict:
            sample['gt']['data'], sample['gt']['ksp'] = coilCombiner(
                sample['gt']['data'], sample['gt']['ksp'], self.combine_coils['fullyI'], self.combine_coils['fullyK'])
            if 'inp' in sample:
                sample['inp']['data'], sample['inp']['ksp'] = coilCombiner(
                    sample['inp']['data'], sample['inp']['ksp'], self.combine_coils['underI'], self.combine_coils['underK'])

        if bool(self.complex_image_modes) and type(self.complex_image_modes) is dict:
            sample['gt']['data'] = complex_modeconverter(
                sample['gt']['data'], mode=self.complex_image_modes["fullyI"], channel_dim=not self.combine_coils['fullyI'])
            sample['gt']['ksp'] = complex_modeconverter(
                sample['gt']['ksp'], mode=self.complex_image_modes["fullyK"], channel_dim=not self.combine_coils['fullyK'])
            if 'inp' in sample:
                sample['inp']['data'] = complex_modeconverter(
                    sample['inp']['data'], mode=self.complex_image_modes["underI"], channel_dim=not self.combine_coils['underI'])
                sample['inp']['ksp'] = complex_modeconverter(
                    sample['inp']['ksp'], mode=self.complex_image_modes["underK"], channel_dim=not self.combine_coils['underK'])

        if bool(self.custom_transform):
            sample = self.custom_transform(sample)

        # TODO: currently it doesn't take into account the change of complex modes.
        if self.expand_ch:
            sample['gt']['data'] = sample['gt']['data'].unsqueeze(
                0) if self.combine_coils['fullyI'] else sample['gt']['data']
            sample['gt']['ksp'] = sample['gt']['ksp'].unsqueeze(
                0) if self.combine_coils['fullyK'] else sample['gt']['ksp']
            if 'inp' in sample:
                sample['inp']['data'] = sample['inp']['data'].unsqueeze(
                    0) if self.combine_coils['underI'] else sample['inp']['data']
                sample['inp']['ksp'] = sample['inp']['ksp'].unsqueeze(
                    0) if self.combine_coils['underK'] else sample['inp']['ksp']

        return sample


def createFastMRIDS(root_gt: Union[str, Sequence[str]] = "/mnt/BMMR/data/Soumick/ChallangeDSs/fastMRI/Knee/multicoil_train/multicoil_train",
                    root_input: Optional[Union[str, Sequence[str]]] = None,
                    filename_filter: Optional[Union[str,
                                                    Sequence[str]]] = None,
                    processed_csv: str = "",
                    split_csv: str = "",
                    split: str = "",
                    init_transforms: Optional[Callable] = None,
                    aug_transforms: Optional[Callable] = None,
                    transforms: Optional[Callable] = None,
                    expand_ch=True,
                    combine_coils = {"underK": False, "fullyK": False, "underI": False, "fullyI": False},
                    complex_image_modes = {"underI": 1, "fullyI": 1, "underK": 0, "fullyK": 0}, # 0: complex image, 1: magnitude image, 2: real image, 3: channel-wise mag+phase [-3 to invert], 4: channel-wise real+imag [-4 to invert]. # for ksp, only 0, 3, 4 are valid - along with the iverses
                    fastMRI_challenge="multicoil",
                    use_dataset_cache=True,
                    sample_rate=None,
                    volume_sample_rate=None,
                    num_cols=None,
                    mask_type="random",
                    center_fractions=[0.08],
                    accelerations=[4]):

    assert not bool(root_input), "root_input should be None, as for now, fastMRI only uses on-the-fly operations to create input"
    assert not bool(init_transforms) and not bool(aug_transforms) and not bool(transforms), "all three types of transforms should be None, as for now, fastMRI DS doesn't support them"

    mask_func = create_mask_for_mask_type(
        mask_type_str=mask_type, center_fractions=center_fractions, accelerations=accelerations)
    dataprep = fastMRIDataPrep(
        fastMRI_challenge, mask_func=mask_func, use_seed=True)
    
    dataset = fastMRIDS(root=root_gt, challenge=fastMRI_challenge,
                        dataprep=dataprep, use_dataset_cache=use_dataset_cache,
                        sample_rate=sample_rate, volume_sample_rate=volume_sample_rate,
                        dataset_cache_file=f"{os.path.dirname(root_gt)}/dataset_cache_{os.path.basename(root_gt)}.pkl", num_cols=num_cols,
                        transform=None, expand_ch=expand_ch, combine_coils=combine_coils, complex_image_modes=complex_image_modes)

    if not bool(processed_csv) or not os.path.isfile(processed_csv):
        if bool(filename_filter):
            if type(filename_filter) is str:
                filename_filter = [filename_filter]
            dataset.examples = [ex for ex in dataset.examples if any(
                filt in str(ex[0]) for filt in filename_filter)]

        if bool(split_csv):
            df = pd.read_csv(split_csv)[split]
            df.dropna(inplace=True)
            df = list(df)
            dataset.examples = [ex for ex in dataset.examples if any(d in str(ex[0]) for d in df)]

        data_df = pd.DataFrame(dataset.examples, columns=["fname", "dataslice", "metadata"])

        if bool(processed_csv):
            data_df.to_csv(processed_csv)
    else:
        data_df = pd.read_csv(processed_csv)
        data_df = data_df[["fname", "dataslice", "metadata"]]
        ex = [tuple(x) for x in data_df.to_numpy()]
        ex = [(Path(x[0]), x[1], ast.literal_eval(x[2])) for x in ex]
        dataset.examples = ex

    filenames = data_df.fname.unique()
    if type(filenames[0]) is str:
        filenames = [os.path.basename(f) for f in filenames]
    else:
        filenames = [f.name for f in filenames]

    return dataset, filenames
