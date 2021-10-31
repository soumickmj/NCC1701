datfile=r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Skyra\20190313\RAW\4 - Mario Brain\meas_MID00090_FID13193_sv_ga_brain_384sp_nodix.dat"


# Attempt8-IXIT1HHVarden1D15-Resnet2Dv214PReLU-SSIMLoss-AllSlices

import h5py
import numpy as np
filepath = r'D:\ROugh\sv_ga_brain_384sp_nodix.mat'
arrays = {}
f = h5py.File(filepath)
fullrawdata = np.array(f['fullrawdata']).transpose()
dataDims = np.array(f['dataDims']).transpose()
imageresolution = f['imageresolution']
trajectory = f['trajectory']

print('test')




#import SimpleITK as sitk

#####Perform registration
#fixedImage = sitk.ReadImage(r'D:\temp\hd\12fe.nii.gz') #low
#movingImage = sitk.ReadImage(r'D:\temp\hd\07fi.nii.gz') #low


#try:
#    elastixImageFilter = sitk.ElastixImageFilter()
#except:
#    elastixImageFilter = sitk.SimpleElastix()

#elastixImageFilter.SetFixedImage(fixedImage)
#elastixImageFilter.SetMovingImage(movingImage)

#parameterMapVector = sitk.VectorOfParameterMap()
#parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
#elastixImageFilter.SetParameterMap(parameterMapVector)

#elastixImageFilter.Execute()
#resultImage = elastixImageFilter.GetResultImage() #output image
#transformParameterMap = elastixImageFilter.GetTransformParameterMap() #deform field like thing
#sitk.WriteImage(resultImage, r'D:\temp\hd\moved07ldfe.nii.gz')

##########apply deform to high
#fixedImageHD = sitk.ReadImage(r'D:\temp\hd\12fe.nii.gz') #high
#movingImageHD = sitk.ReadImage(r'D:\temp\hd\07fi.nii.gz') #high

#try:
#    transformixImageFilter = sitk.TransformixImageFilter()
#except:
#    transformixImageFilter = sitk.SimpleTransformix()

#transformixImageFilter.SetTransformParameterMap(transformParameterMap)

#transformixImageFilter.SetMovingImage(movingImageHD)
#transformixImageFilter.Execute()
#resultImageHD = transformixImageFilter.GetResultImage()
#sitk.WriteImage(resultImageHD, r'D:\temp\hd\moved07HDfe.nii.gz')