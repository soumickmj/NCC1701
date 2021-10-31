import numpy as np
import sigpy as sp
from sigpy.mri import app as MRIApps

class Controller4Conventional(object):
    def __init__(self, device, domain):
        self.device = device
        self.IsCuda = True if self.device > -1 else False
        self.domain = domain

    def RecoCalls(self, under, cmaps=None):
        if self.domain == 'hartley':
            per #Perform Hartley to Fourier
        elif self.domain == 'image':
            per #Perform Image to Foutier
        
        if cmaps is None:
            cmaps = MRIApps.EspiritCalib(under, device=self.device).run() #Calculate sensitivity if not supplied

        #SENSE Recon
        lamda = 0.01 #Lambda value from official tutorial of sigpy
        img_sense = MRIApps.SenseRecon(under, cmaps, lamda=lamda, device=self.device).run() #Lambda optional. Returns coil-combined complex

        #L1 Wavelet Regularized Reconstruction
        lamda = 0.005 #Lambda value from official tutorial of sigpy
        img_l1wav = MRIApps.L1WaveletRecon(under, cmaps, lamda, device=self.device).run() #Returns coil-combined complex

        #TV Recon
        lamda = 0.005 #Lambda value from official tutorial of sigpy
        img_tv = MRIApps.TotalVariationRecon(under, cmaps, lamda, device=self.device).run() #Returns coil-combined complex

        #JSense Recon
        img_jsense = MRIApps.JsenseRecon(under, device=self.device).run() #Returns non-coil-combined complex (channel-first)
        img_jsense_rss = np.sum(np.abs(img_jsense)**2, axis=0)**0.5 #Returns coil-combined non-complex

        return {'SENSE':img_sense, 'L1Wavelet':img_l1wav, 'TV':img_sense, 'jSENSE':img_jsense_rss}


    def Test(self, dataloader):
         for i, batch in enumerate(dataloader, 0):
             try:
                 for data in batch: #If data is provided with a batch_size more than 1
                    fully = data['fully']
                    under = data['under']

                    if self.domain == 'image': #If the data is in Image space, then need to undersample again from the fully sampled
                        per #undersample
                    else:
                        per #convert fully to image

                    if(self.IsCuda):
                        under = sp.to_device(under, device=self.device)

                    if len(fully.shape) == 4: #3D data supplied, functions to be applied on 2D basis
                        reocnSlc = []
                        for slc in under:
                            reocnSlc.append(self.RecoCalls(slc))
                        generated = {k: v for d in L for k, v in d.items()} #solve it TODO
                    else:
                        generated = self.RecoCalls(under)
                
                    if(self.IsCuda):
                        under = sp.to_device(under, device=sp.Device(-1))
                        generated = sp.to_device(generated, device=sp.Device(-1))

                    #Deal with batch TODO
                    for j in range(0, len(data['subjectName'])): 
                        path4output = os.path.join(output_path, data['subjectName'][j], data['fileName'][j]) 
                        if not self.NetType == 3:
                            if self.NetType == 2:
                                doComparisons=False
                            else:
                                doComparisons=True
                            if not convertedToDomain:
                                generatedImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(generated.data[j]))
                                fullyImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(fully[j]))
                                underImg = self.helpOBJ.Domain2Image(transformTensorToNumpy(under[j]))
                            else:
                                generatedImg = transformTensorToNumpy(generated.data[j])
                                fullyImg = transformTensorToNumpy(fully[j])
                                underImg = transformTensorToNumpy(under[j])
                            ValidateNStoreRecons(generatedImg,fullyImg,underImg,path4output,doComparisons=doComparisons)
                        else:
                            fastMRIReconStore(generated.data[j],fully[j],under[j],data['mask'][j],path4output,
                                                getSSIMMap=True,channelSum=True,domain=self.domain,use_data_consistency=True,roi_crop_res=None,save_raw=True)

                