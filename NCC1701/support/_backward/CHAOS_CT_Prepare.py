# encoding: utf-8

__author__ = 'Soumick Chatterjee'

import os
import re
import numpy as np
import PIL.Image
import argparse
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk
from collections import defaultdict

MAPPING = {
    0: (0, 'background'),
    255: (1, 'liver'),
    160: (2, 'right_kidney'),
    240: (3, 'left_kidney'),
    255: (4, 'spleen')
}

def get_patients(path):
    patients = []
    regex = '^\d+$'
    for x in os.listdir(path):
        if re.match(regex, x):
            patients.append(x)
    return patients

def get_images_from_patient(patient_path):
    all_images = []
    all_masks = []
    sequence_type = 'CT'
    dicoms = os.path.join(patient_path, 'DICOM_anon')
    dcm_images = glob(os.path.join(dicoms, '*.dcm'))
    gt = os.path.join(patient_path, 'Ground')
    gt_images = glob(os.path.join(gt, '*.png'))
    gt_images = sorted(gt_images, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.png')[0]))

    # Try to read the dicom images
    slice_thicknesses = []

    images_dict = defaultdict(list)
    location_dict = {}
    for dcm in dcm_images:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(dcm)
        file_reader.ReadImageInformation()
        slice_thickness = float(file_reader.GetMetaData('0018|0050').strip())
        slice_thicknesses.append(slice_thickness)
        slice_location = float(file_reader.GetMetaData('0020|1041').strip())
        echo_time = 0
        images_dict[echo_time].append(dcm)
        location_dict[dcm] = slice_location

    assert len(set(slice_thicknesses)) == 1, 'Multiple thicknesses in images'

    for echo_time, dcm_fns in images_dict.items():
        dcm_fns = sorted(dcm_fns, key=lambda x: location_dict[x])
        slices = [sitk.ReadImage(_) for _ in dcm_fns]
        vol_img = sitk.Image(slices[0].GetSize()[0], slices[0].GetSize()[1], len(slices), slices[0].GetPixelID())
        for idx_z, slice_vol in enumerate(slices):
            vol_img = sitk.Paste(vol_img, slice_vol, slice_vol.GetSize(), destinationIndex=[0, 0, idx_z])
        vol_img.SetSpacing(slices[0].GetSpacing())
        vol_img.SetOrigin(slices[0].GetOrigin())
        all_images.append((sequence_type, echo_time, vol_img))

    # Work with masks. They are the same for each sequence.
    gt_arr = np.stack([np.asarray(PIL.Image.open(_)) for _ in gt_images])
    gt_mask = np.zeros_like(gt_arr).astype(np.uint8)
    for png_val, (arr_key, label_name) in MAPPING.items():
        gt_mask[gt_arr == png_val] = arr_key
    gt_sitk_mask = sitk.GetImageFromArray(gt_mask)
    gt_sitk_mask.SetOrigin(vol_img.GetOrigin())
    gt_sitk_mask.SetDirection(vol_img.GetDirection())
    gt_sitk_mask.SetSpacing(vol_img.GetSpacing())
    all_masks.append((sequence_type, gt_sitk_mask))

    return all_images, all_masks

def main():

    root_dir = r'D:\Datasets\CHAOS_dataset\CT'
    output_dir_nii = r'D:\Datasets\CHAOS_dataset\CT_NII'

    patients = get_patients(root_dir)

    for patient in tqdm(patients):
        images, masks = get_images_from_patient(os.path.join(root_dir, patient))
        images.sort(key=lambda x: x[1])  # Sort on echo time, longer echo time is the in-phase image

        for idx, image_list in enumerate(images):
            sequence_type, echo_time, image = image_list
            fn = f'CT_image.nii'

            write_to_folder = os.path.join(output_dir_nii, f'Patient_{patient}')
            os.makedirs(write_to_folder, exist_ok=True)
            sitk.WriteImage(image, os.path.join(write_to_folder, fn), True)

        for mask_name, mask in masks:
            fn = f'{mask_name}_mask.nii'
            write_to_folder = os.path.join(output_dir_nii, f'Patient_{patient}')
            sitk.WriteImage(mask, os.path.join(write_to_folder, fn), True)

if __name__ == '__main__':
    main()