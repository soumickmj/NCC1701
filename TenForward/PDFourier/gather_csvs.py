import pandas as pd
from glob import glob
from tqdm import tqdm

# Step 1 of 

result_root = "/run/media/soumick/Enterprise/Results/PDFourier"
res_file = "Results"
# res_file = "Results2D"
# res_file = "Results2DSkip1"
# res_file = "Results2DSkip1"
res_file = "Results3DSkip2"
# task = "MoCo" #MoCo or ReCo
task = "ReCo" #MoCo or ReCo

dataset = "fastMRI_AXT2_16Coil_768x396_acc16_cen0p08"
# dataset = "T1IXI-ax-1mm-iso"

result_root += "/" + task
res_csvs = glob(f"{result_root}/**/{dataset}_*/{res_file}.csv", recursive=True)

dfs = []
for csv in tqdm(res_csvs):
    model_meta = csv.split(result_root)[1].split(dataset+"_")    
    modelTrainID = model_meta[-1].split("/"+res_file+".csv")[0]
    trainID = f"{dataset}_{modelTrainID}"
    model_meta = "_".join(list(filter(None, model_meta[0].split("/"))))
    df = pd.read_csv(csv)
    df["ModelMeta"] = model_meta
    df["trainID"] = trainID
    df["modelTrainID"] = modelTrainID
    df["ResPath"] = csv.split(".csv")[0]
    dfs.append(df)
df = pd.concat(dfs)
df.to_csv(f"{result_root}/00ResAnalysis/{res_file}_{dataset}.csv")




