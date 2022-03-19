
import os
from Engineering.utilities import calc_metircs
from glob import glob
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import numpy as np


def gen3DFiltRes(root):
    res_folders = glob(f"{root}/**/Results.csv", recursive=True)

    for rf in res_folders:
        folder = rf.replace(".csv", "")
        vols = glob(f"{folder}/**")
        test_metrics = []
        try:
            for vol in tqdm(vols):
                inp = np.array(nib.load(f"{vol}/inp.nii.gz").get_fdata())[...,:-2]
                gt = np.array(nib.load(f"{vol}/gt.nii.gz").get_fdata())[...,:-2]
                out = np.array(nib.load(f"{vol}/out.nii.gz").get_fdata())[...,:-2]
                try:
                    outCorrected = np.array(nib.load(f"{vol}/outCorrected.nii.gz").get_fdata())[...,:-2]
                except:
                    outCorrected = None

                
                inp_metrics, inp_ssimMAP, inp_diff = calc_metircs(gt, inp, tag="Inp", norm4diff=True)
                out_metrics, out_ssimMAP, out_diff = calc_metircs(gt, out, tag="Out", norm4diff=True)
                metrics = {**out_metrics, **inp_metrics}

                if outCorrected is not None:
                    outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(gt, outCorrected, tag="OutCorrected", norm4diff=True)
                    metrics = {**metrics, **outCorrected_metrics}
                
                metrics['file'] = os.path.basename(vol)
                test_metrics.append(metrics)

            df = pd.DataFrame.from_dict(test_metrics)
            df.to_csv(rf.replace(".csv", "3DSkip2.csv"), index=False)
                        
            # df1 = pd.DataFrame.from_dict(test_metrics_filt1)
            # df1.to_csv(rf.replace(".csv", "2DSkip1.csv"), index=False)
                        
            # df2 = pd.DataFrame.from_dict(test_metrics_filt2)
            # df2.to_csv(rf.replace(".csv", "2DSkip2.csv"), index=False)
        except:
            pass