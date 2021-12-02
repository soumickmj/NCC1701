import random
from scipy import ndimage
import numpy as np
import multiprocessing.dummy as multiprocessing

from Engineering.transforms.transforms import SuperTransformer

class Motion2Dv0(SuperTransformer):
    def __init__(self, sigma_range=(0.10, 2.5), n_threads=10, **kwargs):
        kwargs['gt2inp'] = True
        super().__init__(**kwargs)
        if type(sigma_range) == str:
            sigma_range = tuple([float(tmp) for tmp in sigma_range.split(",")])
        self.sigma_range = sigma_range
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        rot = self.sigma*random.randint(-1,1)
        img_aux = ndimage.rotate(self.img, rot, reshape=False)
        img_h = np.fft.fft2(img_aux)
        if self.axis_selection == 0:
            self.aux[:,idx]=img_h[:,idx]
        else:
            self.aux[idx,:]=img_h[idx,:]

    def apply(self, img):
        self.img = img
        self.aux = np.zeros(img.shape, dtype=img.dtype) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]))
        else:
            for idx in range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]):
                self.__perform_singlePE(idx)
        cor = np.abs(np.fft.ifft2(self.aux)).astype(img.dtype) 
        del self.img, self.aux, self.axis_selection, self.sigma
        return (cor-cor.min())/(cor.max()-cor.min()+np.finfo(np.float32).eps) 

class Motion2Dv1(SuperTransformer):
    def __init__(self, sigma_range=(0.10, 2.5), restore_original=0, n_threads=20, **kwargs):
        kwargs['gt2inp'] = True
        super().__init__(**kwargs)
        if type(sigma_range) == str:
            sigma_range = tuple([float(tmp) for tmp in sigma_range.split(",")])
        self.sigma_range = sigma_range
        self.restore_original = restore_original
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        img_aux = ndimage.rotate(self.img, self.random_rots[idx], reshape=False)
        img_h = np.fft.fft2(img_aux)            
        if self.axis_selection == 0:
            self.aux[:,self.portion[idx]]=img_h[:,self.portion[idx]]  
        else:
            self.aux[self.portion[idx],:]=img_h[self.portion[idx],:]  

    def apply(self, img):
        self.img = img
        self.aux = np.zeros(img.shape, dtype=img.dtype) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]

        if self.axis_selection == 0:
            dim = 1
        else:
            dim = 0

        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[dim]//n_), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[dim]//n_), 
                                                        high=int(img.shape[dim]), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int))) 
            self.portion = np.concatenate((portiona, portionb))  
        else:
            self.portion = np.sort(np.unique(np.random.randint(low=int(img.shape[dim]//2)-int(img.shape[dim]//n_+1), 
                                                     high=int(img.shape[dim]//2)+int(img.shape[dim]//n_+1), 
                                                     size=int(img.shape[dim]//n_+1), dtype=int)))
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        self.random_rots = self.sigma * np.random.randint(-1,1,len(self.portion))

        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(len(self.portion)-1))
        else:
            for idx in range(len(self.portion)-1):
                self.__perform_singlePE(idx)     
        cor = (np.abs(np.fft.ifft2(self.aux)) + (self.restore_original * img)).astype(img.dtype) 

        del self.img, self.aux, self.axis_selection, self.portion, self.random_rots
        return (cor-cor.min())/(cor.max()-cor.min()+np.finfo(np.float32).eps) 