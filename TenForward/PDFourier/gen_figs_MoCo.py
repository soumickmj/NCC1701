import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Step 4 of 

sns.set_theme(style="darkgrid")

result_root = "/run/media/soumick/Enterprise/Results/PDFourier"
res_type = "Results"
res_type = "Results2D"
# res_type = "Results2DSkip1"
# res_type = "Results2DSkip2"
# res_type = "Results3DSkip2"

dataset = "T1IXI-ax-1mm-iso"

result_root += "/" + "MoCo"
df = pd.read_csv(f"{result_root}/00ResAnalysis/{res_type}_{dataset}.csv")

models = {
    "Corrupted": "Corrupted",
    "ResNet14L1-Mot1t2-rnd50": "ReconResNet",
    "PDOrigMagPrimCReLUL1-Mot1t2-rnd50": "Fourier-PD",
    # "PDUNetMagPrimCReLU10ITL1-Mot1t2-rnd50": "Fourier-PDUNet",
    "PDUNetMagPrimCReLUL1-Mot1t2-rnd50": "Fourier-PDUNet",
}
legend_order = ["Corrupted", "ReconResNet", "Fourier-PD", "Fourier-PDUNet"]

model_keys = list(models.keys())

df = df[df.modelTrainID.isin(models.keys())]

med_pdu = df[df.modelTrainID==model_keys[-1]].median()['SSIMOut']
med_pdorig = df[df.modelTrainID==model_keys[-2]].median()['SSIMOut']
med_res = df[df.modelTrainID==model_keys[-3]].median()['SSIMOut']


sortlisted_files_pdunet = list(df[(df.modelTrainID==model_keys[-1]) & (df["SSIMOut"].round(3)==round(med_pdu,3))].sort_values("SSIMOut", ascending=False).file)
sortlisted_files_pdorig = list(df[(df.modelTrainID==model_keys[-2]) & (df["SSIMOut"].round(3)==round(med_pdorig,3))].sort_values("SSIMOut", ascending=False).file)
sortlisted_files_resnet = list(df[(df.modelTrainID==model_keys[-3]) & (df["SSIMOut"].round(3)==round(med_res,3))].sort_values("SSIMOut", ascending=False).file)
sortlisted_files_common = list(set(sortlisted_files_pdunet).intersection(sortlisted_files_pdorig).intersection(sortlisted_files_resnet))

for f in sortlisted_files_pdunet:
    dfFile = df[df.file == f]
    for m in model_keys[1:]:
        p = dfFile[dfFile.modelTrainID==m].ResPath.iloc[0]+"/"+f.split(".")[0]
        p_inp = f"{p}/inp.nii.gz"
        p_inp_ssim = f"{p}/ssimMAPInp.nii.gz"
        p_inp_diff = f"{p}/diffInp.nii.gz"

        p_out = f"{p}/Out.nii.gz"
        p_ssim = f"{p}/ssimMAPOut.nii.gz"
        p_diff = f"{p}/diffOut.nii.gz"


plotnow(df, "SSIM", result_root, dataset, legend_order, x="Method")
plotnow(df, "NRMSE", result_root, dataset, legend_order, x="Method")
plotnow(df, "PSNR", result_root, dataset, legend_order, x="Method")