import numpy as np
import sigpy as sp
from sigpy.mri import app as MRIApps

#####################For CS Reco on Cartesian Data


#ksp = np.load(r"D:\ROugh\sigpydata\cartesian_ksp.npy") #Data should be channel-first complex

#device = sp.Device(0) #id = -1 represents CPU, and others represents the id_th GPUs.
#if device.id > -1:
#    ksp = sp.to_device(ksp, device=device)

##Calculate sensitivity
#mps = MRIApps.EspiritCalib(ksp, device=device).run()

##SENSE Recon
#lamda = 0.01 #Lambda value from official tutorial of sigpy
#img_sense = MRIApps.SenseRecon(ksp, mps, lamda=lamda, device=device).run() #Lambda optional. Returns coil-combined complex

##L1 Wavelet Regularized Reconstruction
#lamda = 0.005 #Lambda value from official tutorial of sigpy
#img_l1wav = MRIApps.L1WaveletRecon(ksp, mps, lamda, device=device).run() #Returns coil-combined complex

##TV Recon
#lamda = 0.005 #Lambda value from official tutorial of sigpy
#img_tv = MRIApps.TotalVariationRecon(ksp, mps, lamda, device=device).run() #Returns coil-combined complex

##JSense Recon
#img_jsense = MRIApps.JsenseRecon(ksp, device=device).run() #Returns non-coil-combined complex (channel-first)

#####################For CS Reco on Cartesian Data <END>


#####################For Gridding Reco on Radial Data


ksp = np.load(r'D:\ROugh\sigpydata\projection_ksp.npy') #Data should be channel-first complex. Eg: 12x96x512, where n_coils = 12, n_spokes = 96, two times oversampled raw data (n_datapoints) = 512
coord = np.load(r'D:\ROugh\sigpydata\projection_coord.npy') #Float data Eg: 96x512x2, where n_spokes = 96, n_datapoints = 512, 2 =?
dcf = (coord[..., 0]**2 + coord[..., 1]**2)**0.5 #Float data Eg: 96x512, where n_spokes = 96, n_datapoints = 512

device = sp.Device(0) #id = -1 represents CPU, and others represents the id_th GPUs.
if device.id > -1:
    ksp = sp.to_device(ksp, device=device) 
    coord = sp.to_device(coord, device=device) 
    dcf = sp.to_device(dcf, device=device) 

#Perform adjoin NUFFT
img_grid = sp.nufft_adjoint(ksp * dcf, coord) #Returns non-coil-combined complex (channel-first)

#Coil combine: Root-sum-of-squares
img_rss = np.sum(np.abs(img_grid)**2, axis=0)**0.5 #Returns coil-combined non-complex

#Tom make the gridding faster, we can specify oversampling ratio and kernel width
img_grid_tune = sp.nufft_adjoint(ksp * dcf, coord, oversamp=1, width=2) #Returns non-coil-combined complex (channel-first)
img_rss_tune = np.sum(np.abs(img_grid_tune)**2, axis=0)**0.5 #Returns coil-combined non-complex
#####################For CS Reco on Cartesian Data <END>

#######Some additonal examples:-
#sigpy.nufft(input, coord)
#sigpy.to_pytorch(array, requires_grad=True)
#sigpy.from_pytorch(tensor, iscomplex=False)
#sigpy.to_pytorch_function(linop, input_iscomplex=False, output_iscomplex=False)
#sigpy.fwt(input, wave_name='db4', axes=None, level=None, apply_zpad=True)
#sigpy.iwt(input, oshape, wave_name='db4', axes=None, level=None, inplace=False)
###Most of those functions are present as operators as well: https://sigpy.readthedocs.io/en/latest/core_linop.html
#sigpy.mri.poisson(img_shape, accel, K=30, calib=[0, 0], crop_corner=True, return_density=False, seed=0) #accel – Target acceleration factor. Greater than 1. K  – maximum number of samples to reject.
#sigpy.mri.radial(coord_shape, img_shape, golden=True) #coord_shape – coordinates of shape [ntr, nro, ndim], where ntr is the number of TRs, nro is the number of readout, and ndim is the number of dimensions. 
#sigpy.mri.spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm, gamma=267800000.0) #FOV in Meters, N = shape, f_sampling = undersampling factor in freq encoding direction, R = undersamp factor, ninterleaves=number of spiral interleaves, gm (float) – maximum gradient amplitude (T/m), sm (float) – maximum slew rate (T/m/s), gamma (float) – gyromagnetic ratio in rad/T/s    
#sigpy.mri.birdcage_maps(shape, r=1.5, nzz=8) #Simulates birdcage coil sensitivies. Params: shape – sensitivity maps shape, can be of length 3, and 4., r – relative radius of birdcage. nzz – number of coils per ring.
#sigpy.mri.get_cov(noise) #Get covariance matrix from noise measurements. Input: [num_coils, …] Returns: [num_coils x num_coils]
#sigpy.mri.whiten(ksp, cov) #Whitens k-space measurements (channel-first).