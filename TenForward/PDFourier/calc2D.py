
import os
from Engineering.utilities import calc_metircs
from glob import glob
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np


def gen2DRes(root):
    res_folders = glob(f"{root}/**/Results.csv", recursive=True)

    test_metrics_filt1 = []
    test_metrics_filt2 = []
    for rf in res_folders:
        folder = rf.replace(".csv", "")
        vols = glob(f"{folder}/**")
        test_metrics = []
        try:
            for vol in tqdm(vols):
                inp = np.array(nib.load(f"{vol}/inp.nii.gz").get_fdata())
                gt = np.array(nib.load(f"{vol}/gt.nii.gz").get_fdata())
                out = np.array(nib.load(f"{vol}/out.nii.gz").get_fdata())
                try:
                    outCorrected = np.array(nib.load(f"{vol}/outCorrected.nii.gz").get_fdata())
                except:
                    outCorrected = None

                for i in range(gt.shape[-1]):
                    inp_metrics, inp_ssimMAP, inp_diff = calc_metircs(gt[...,i], inp[...,i], tag="Inp", norm4diff=True)
                    out_metrics, out_ssimMAP, out_diff = calc_metircs(gt[...,i], out[...,i], tag="Out", norm4diff=True)
                    metrics = {**out_metrics, **inp_metrics}

                    if outCorrected is not None:
                        outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(gt[...,i], outCorrected[...,i], tag="OutCorrected", norm4diff=True)
                        metrics = {**metrics, **outCorrected_metrics}

                    metrics['file'] = os.path.basename(vol)
                    metrics['sliceID'] = i
                    test_metrics.append(metrics)

                    # if i+1 < gt.shape[-1]:
                    #     test_metrics_filt1.append(metrics)

                    # if i+2 < gt.shape[-1]:
                    #     test_metrics_filt2.append(metrics)

            df = pd.DataFrame.from_dict(test_metrics)
            df.to_csv(rf.replace(".csv", "2D.csv"), index=False)

            # df1 = pd.DataFrame.from_dict(test_metrics_filt1)
            # df1.to_csv(rf.replace(".csv", "2DSkip1.csv"), index=False)

            # df2 = pd.DataFrame.from_dict(test_metrics_filt2)
            # df2.to_csv(rf.replace(".csv", "2DSkip2.csv"), index=False)
        except:
            pass