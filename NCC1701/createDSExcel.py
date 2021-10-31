from utils.TorchDS.IXI import MRITorchDS as dsClass

folder_path_fully = r'E:\Datasets\IXI\IXI-T1'
folder_path_under = r'E:\Datasets\IXI\Under\IXI-T1\1DVarden30Mask'
extension_under = None
filename_filter = ['Guys','HH']
xlsx_path = r'E:\Datasets\IXI\guysNhh_split100.xlsx'
nTrain=100
nTest=100
nVal=100


#Useless params - don't change values though
domain = 'image'
transform = None
getROIMode = False
undersampling_mask = None

ds = dsClass(folder_path_fully,folder_path_under,extension_under,domain=domain,transform=transform,getROIMode=getROIMode,undersampling_mask=undersampling_mask,filename_filter=filename_filter)
ds.GenerateTrainTestSet(xlsx_path, nTrain=nTrain, nTest=nTest, nVal=nVal)
