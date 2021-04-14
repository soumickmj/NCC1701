import numpy as np
import scipy.io as sio
import sigpy as sp
import sigpy.mri as mr
from pynufft import NUFFT as NUFFT_cpu
from Math.FrequencyTransforms import fft2c, ifft2c

def Hendrik_PROSIT(maskORom_path, fully_img=None, simulate_coil=4):
    # N = 256  # image size
    ImgShape = fully_img.shape
    NCoils = 4  # number of coils
    KspShape = (NCoils, *ImgShape)
    Acc = 16  # undersampling factor

    # create coil-wise k-space
    Maps = mr.birdcage_maps(KspShape)
    Ksp = sp.fft(fully_img * Maps)

    # create undersampled k-space
    Mask = mr.poisson(ImgShape, 16, 30, (24, 24))
    Mask = sio.loadmat(maskORom_path)['mask']
    KspUnder = Ksp * Mask

    # estimate maps using ESPIRIT from the undersampled k-space
    EMaps = mr.app.EspiritCalib(KspUnder).run()

    # set-up LinOps
    F = sp.linop.FFT(KspShape, axes=(-1, -2))
    S = sp.linop.Multiply(ImgShape, EMaps)
    P = sp.linop.Multiply(KspUnder.shape, Mask)
    A = P * F * S

    # noisy fully sampled image
    pGrd = S.H * F.H * Ksp

    Lambda = 0.01
    MaxIter = 30

    rGrd = sp.app.LinearLeastSquares(A, KspUnder, alpha=1, max_iter=MaxIter, max_power_iter=1,
                                        proxg=sp.prox.L2Reg(ImgShape, lamda=Lambda)).run()


def Undersamp(maskORom_path, fully_ksp=None, fully_img=None, simulate_coil=4): 
    temp_mat =  sio.loadmat(maskORom_path)
    if simulate_coil > 1:
        if fully_img is None:
            # fully_img = ifft2c(fully_ksp, axes=(-1,-2), shiftAxes = (-1,-2))
            fully_img = sp.ifft(fully_ksp)
        mps = mr.birdcage_maps((simulate_coil, *fully_img.shape))
        fully_img = mps * fully_img
    else:
        mps = None

    if "mask" in temp_mat:
        mask = temp_mat['mask']   

        if fully_ksp is None:
            # fully_ksp = fft2c(fully_img, axes=(-1,-2), shiftAxes = (-1,-2))
            fully_ksp = sp.fft(fully_img)

        under_ksp = fully_ksp * mask
        # under_img = ifft2c(under_ksp, axes=(-1,-2), shiftAxes = (-1,-2))
        under_img = sp.ifft(under_ksp, axes=(-1,-2))
    else:
        if fully_img is None:
            # fully_img = ifft2c(fully_ksp, axes=(-1,-2), shiftAxes = (-1,-2))
            fully_img = sp.ifft(fully_ksp)

        om = temp_mat['om']
        dcf = temp_mat['dcf'].squeeze()
        NufftObjOM = NUFFT_cpu()
        imageSize=fully_img.shape[-1]
        baseresolution = imageSize*2
        interpolationSize4NUFFT = 6
        Nd = (baseresolution, baseresolution)# image size
        Kd = (baseresolution*2, baseresolution*2)  # k-space size - TODO: multiply back by 2
        Jd = (interpolationSize4NUFFT, interpolationSize4NUFFT)  # interpolation size
        NufftObjOM.plan(om, Nd, Kd, Jd)

        under_imgs = []
        if simulate_coil == 1:
            fully_img = np.expand_dims(fully_img, 0)
        for i in range(simulate_coil):
            oversam_fully = np.zeros((baseresolution,baseresolution), dtype=fully_img.dtype)
            oversam_fully[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = fully_img[i]
            yUnder = NufftObjOM.forward(oversam_fully)
            y = np.multiply(dcf,yUnder)
            oversam_under = NufftObjOM.adjoint(y)
            under_imgs.append(oversam_under[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2])
        under_img = np.array(under_imgs).squeeze()

        # under_ksp = fft2c(under_img, axes=(-1,-2), shiftAxes = (-1,-2))
        under_ksp = sp.fft(under_img)

    return under_ksp, under_img, mps



def CSReco(under_ksp, channel_expand=False, mode=2, mps=None):
    lamda2 = 0.01
    if mps is None:
        if channel_expand:
            under_ksp = np.expand_dims(under_ksp, 0)
            mps = np.ones(under_ksp.shape) + 0j
        else:
            mps = mr.app.EspiritCalib(under_ksp).run()
    if mode==0:
        recon = mr.app.SenseRecon(under_ksp, mps, lamda=lamda2).run()
    elif mode==1:
        recon = mr.app.L1WaveletRecon(under_ksp, mps, lamda=lamda2).run()
    elif mode==2:
        recon = mr.app.TotalVariationRecon(under_ksp, mps, lamda=lamda2).run()
    return recon
