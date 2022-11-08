import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Step 4 of 

sns.set_theme(style="darkgrid")

result_root = "/run/media/soumick/Enterprise/Results/PDFourier"
task = "ReCo" #MoCo or ReCo

dataset = "fastMRI_AXT2_16Coil_768x396_acc16_cen0p08"
# dataset = "T1IXI-ax-1mm-iso"

result_root += f"/{task}"
df = pd.read_csv(f"{result_root}/00ResAnalysis/{dataset}.csv")

models = {
    "Undersampled": "Undersampled",
    "ResNet14L1": "ReconResNet",
    "PDOrigMagPrimCReLUL1": "Fourier-PD",
    "PDUNetMagPrimCReLU10ITL1": "Fourier-PDUNet",
}
legend_order = ["Undersampled", "ReconResNet", "Fourier-PD", "Fourier-PDUNet"]

model_keys = list(models.keys())

df = df[df.modelTrainID.isin(models.keys())]

med_pdu = df[df.modelTrainID==model_keys[-1]].median()['SSIMOutCorrected']
med_pdorig = df[df.modelTrainID==model_keys[-2]].median()['SSIMOutCorrected']


sortlisted_files_pdunet = list(df[(df.modelTrainID==model_keys[-1]) & (df["SSIMOutCorrected"].round(3)==round(med_pdu,3))].sort_values("SSIMOutCorrected", ascending=False).file)
sortlisted_files_pdorig = list(df[(df.modelTrainID==model_keys[-2]) & (df["SSIMOutCorrected"].round(3)==round(med_pdorig,3))].sort_values("SSIMOutCorrected", ascending=False).file)
sortlisted_files_common = list(set(sortlisted_files_pdunet).intersection(sortlisted_files_pdorig))

for f in sortlisted_files_pdunet:
    dfFile = df[df.file == f]
    for m in model_keys[1:]:
        p = dfFile[dfFile.modelTrainID==m].ResPath.iloc[0]+"/"+f.split(".")[0]
        p_inp = f"{p}/inp.nii.gz"
        p_inp_ssim = f"{p}/ssimMAPInp.nii.gz"
        p_inp_diff = f"{p}/diffInp.nii.gz"

        p_out = f"{p}/outCorrected.nii.gz"
        p_ssim = f"{p}/ssimMAPOutCorrected.nii.gz"
        p_diff = f"{p}/diffOutCorrected.nii.gz"


plotnow(df, "SSIM", result_root, dataset, legend_order, x="Method")
plotnow(df, "NRMSE", result_root, dataset, legend_order, x="Method")
plotnow(df, "PSNR", result_root, dataset, legend_order, x="Method")