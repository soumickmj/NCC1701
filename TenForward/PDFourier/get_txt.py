import pandas as pd
from tqdm import tqdm

# Step 2 of 

def groupNwrite(df, groupby, metric, file_obj):
    mean = df.groupby(groupby).mean()[metric]
    median = df.groupby(groupby).median()[metric]
    std = df.groupby(groupby).std()[metric]
    vari = df.groupby(groupby).var()[metric]
    
    file_obj.write("\n============================\n")
    file_obj.write(metric)
    file_obj.write("\n============================\n")

    file_obj.write("\nMean:------------------------------\n")
    file_obj.write(str(mean.round(3).apply(str).str.cat(std.round(3).apply(str), sep="±")))
    file_obj.write("\nMedian:----------------------------\n")
    file_obj.write(str(median.round(3).apply(str).str.cat(vari.round(3).apply(str), sep="±")))
    file_obj.write("\n----------------------------\n")


result_root = "/run/media/soumick/Enterprise/Results/PDFourier"
res_type = "Results"
res_type = "Results2D"
# res_type = "Results2DSkip1"
# res_type = "Results2DSkip2"
res_type = "Results3DSkip2"
task = "ReCo" #MoCo or ReCo
dataset = "fastMRI_AXT2_16Coil_768x396_acc16_cen0p08"
# dataset = "T1IXI-ax-1mm-iso"

result_root += "/" + task
df = pd.read_csv(f"{result_root}/00ResAnalysis/{res_type}_{dataset}.csv")

model_types = df.ModelMeta.unique()
model_types = [i for i in model_types if "notShortlisted" not in i]

with open(f"{result_root}/00ResAnalysis/{res_type}_{dataset}.txt","w") as file_obj:
    groupNwrite(df, "modelTrainID", "SSIMInp", file_obj)
    groupNwrite(df, "modelTrainID", "NRMSEInp", file_obj)
    groupNwrite(df, "modelTrainID", "PSNRInp", file_obj)
    groupNwrite(df, "modelTrainID", "SDofDiffOut", file_obj)
    
    try:
        groupNwrite(df, "modelTrainID", "SSIMOutCorrected", file_obj)
        groupNwrite(df, "modelTrainID", "NRMSEOutCorrected", file_obj)
        groupNwrite(df, "modelTrainID", "PSNROutCorrected", file_obj)
        groupNwrite(df, "modelTrainID", "SDofDiffOutCorrected", file_obj)
    except:
        groupNwrite(df, "modelTrainID", "SSIMOut", file_obj)
        groupNwrite(df, "modelTrainID", "NRMSEOut", file_obj)
        groupNwrite(df, "modelTrainID", "PSNROut", file_obj)
        groupNwrite(df, "modelTrainID", "SDofDiffOut", file_obj)

    try:
        groupNwrite(df, "modelTrainID", "SSIMOutCorrectedReal", file_obj)
        groupNwrite(df, "modelTrainID", "NRMSEOutCorrectedReal", file_obj)
        groupNwrite(df, "modelTrainID", "PSNROutCorrectedReal", file_obj)
        groupNwrite(df, "modelTrainID", "SDofDiffOutCorrectedReal", file_obj)
    except:
        pass


# sortlisted_files_pdunet = {}
# sortlisted_files_pdorig = {}
# sortlisted_files_common = {}
# with open(f"{root_path}/PDUNet_consolidatedSSIMMRINew.txt","w") as file_obj:
#     for con in consolidates:
#         pathparts = con.split("/")
#         runname = f"{pathparts[-3]}_{pathparts[-2]}_{pathparts[-1].split('_')[1].split('.')[0]}"
#         file_obj.write("\n----------------------------\n")
#         file_obj.write(runname)
#         file_obj.write("\n----------------------------\n")

#         df = pd.read_csv(con)
#         mean = df.groupby("Model").mean()["SSIM (Out)"]
#         median = df.groupby("Model").median()["SSIM (Out)"]
#         std = df.groupby("Model").std()["SSIM (Out)"]
        
#         file_obj.write(str(mean.round(3).apply(str).str.cat(std.round(3).apply(str), sep="±")))
#         file_obj.write("\nMedian:\n")
#         file_obj.write(str(median))
#         file_obj.write("\n----------------------------\n")
#         file_obj.write("\n")

#         sortlisted_files_pdunet[runname] = df[(df["Model"]=="Primal-Dual UNet") & (df["SSIM (Out)"].round(3)==round(median['Primal-Dual UNet'],3))].sort_values("SSIM (Out)", ascending=False).Img
#         sortlisted_files_pdorig[runname] = df[(df["Model"]=="Primal-Dual") & (df["SSIM (Out)"].round(3)==round(median['Primal-Dual'],3))].sort_values("SSIM (Out)", ascending=False).Img
#         sortlisted_files_common[runname] = list(set(sortlisted_files_pdunet[runname]).intersection(sortlisted_files_pdorig[runname]))
# pd.DataFrame.from_dict(sortlisted_files_pdunet).to_csv(f"{root_path}/PDUNet_filenames.csv")
# pd.DataFrame.from_dict(sortlisted_files_pdorig).to_csv(f"{root_path}/PD_filenames.csv")
# pd.DataFrame.from_dict(sortlisted_files_common, orient='index').transpose().to_csv(f"{root_path}/PD_PDUNet_filenames.csv")

