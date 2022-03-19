import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
 
# Step 3 (for MoCo) of 

sns.set_theme(style="darkgrid")
# sns.set(font_scale = 1.5)

def convertInp2Out(df, method_name):
    df = df[df.columns.drop(list(df.filter(regex='Out')))]
    df.columns = df.columns.str.replace('Inp', 'Out')
    df.modelTrainID = method_name
    return df

def plotnow(df, y, result_root, res_type, dataset, legend_order, x="Method"):
    fig = plt.figure()
    ax = sns.boxplot(x=x, y=y, data=df, palette=("pastel"), order=legend_order)
    # plt.xticks(rotation=45)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{result_root}/00ResAnalysis/{res_type}_{dataset}_{y}.pdf", format='pdf')
    plt.figure().clear()
    plt.close()

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

dfInp = convertInp2Out(df[df.modelTrainID==model_keys[-1]], "Corrupted")
df = pd.concat([df, dfInp])
df = df[df.modelTrainID.isin(models.keys())]

cols = [l for l in list(df.columns) if ("Out" in l)] 
stat_combos = list(itertools.combinations(model_keys, 2))
with open(f"{result_root}/00ResAnalysis/{res_type}_{dataset}_stat.txt","w") as file_obj:
    for stat_combo  in stat_combos:
        dfA = df[df.modelTrainID==stat_combo[0]]
        dfB = df[df.modelTrainID==stat_combo[1]]
        file_obj.write(f"\n{'_'.join(stat_combo)}:----------------------------\n")
        for c in cols:
            file_obj.write(f"{c}: {str(round(mannwhitneyu(dfA[c], dfB[c]).pvalue, 3))}\n")
print("Stat Done")

df = df.replace({"modelTrainID": models})
df = df.rename(columns={"modelTrainID": "Method", "SSIMOut": "SSIM", "NRMSEOut": "NRMSE", "UQIOut": "UQI", "PSNROut": "PSNR", "SDofDiffOut": "SD of Difference Images"})

plotnow(df, "SSIM", result_root, res_type, dataset, legend_order, x="Method")
plotnow(df, "NRMSE", result_root, res_type, dataset, legend_order, x="Method")
plotnow(df, "PSNR", result_root, res_type, dataset, legend_order, x="Method")

df = df[df.Method != "Corrupted"]
plotnow(df, "UQI", result_root, res_type, dataset, legend_order[1:], x="Method")