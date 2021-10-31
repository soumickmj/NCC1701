import math
import inspect
import numpy as np
import torch
from scipy.stats import multivariate_normal as mvn

class GaussianWeightedLoss(torch.nn.Module):
    """description of class"""

    def __init__(self, data_shape=(256,256), loss_func=torch.nn.MSELoss, sigma4gauss=0.5):
        super(GaussianWeightedLoss, self).__init__()
        if inspect.isclass(loss_func):
            self.loss_func = loss_func()
        else:
            self.loss_func = loss_func
        self.weight = torch.from_numpy(1 - self.gaussianPDF2D(np.zeros(data_shape), sigma4gauss)).float()
        self.weight.requires_grad = False

    def gaussianPDF2D(self, slice, sigma=0.5):
        Xie1 = 5000/(2*math.pi) #Sharpness of the distribution 1, 0 <r <r0
        Xie2 = sigma*1000000/(2*math.pi) #Sharpness of distribution 2, r0 <r <r_max
        s = 0 #diagonal extension of distribution - / +
        r1 = 1 #> = 1, Determines the size of the full coverage of the k-space center, the height of the distribution 1
        r2 = 0.60 #<= 1, influence on the probabilities in the edge of the k-space, height of the distribution 2

        #Fit to mask size
        x1 = np.array(range(slice.shape[0]))
        x2 = np.array(range(slice.shape[1]))     
        X1, X2 = np.meshgrid(x1,x2) 

        #Distribution function 0 <r <r0
        mu1 = np.array([slice.shape[0]//2, slice.shape[1]//2])                         #Average
        #Sigma1 = 1/(2*math.pi)*np.array([[Xie1,Xie1*s], [Xie1*s,Xie1]])         #Covariance matrix, alternative
        Sigma1 = np.array([[Xie1,Xie1*s], [Xie1*s,Xie1]])                     # Covariance matrix
        F1 = (r1*Xie1)*2*math.pi*mvn.pdf(np.array([X1.flatten(), X2.flatten()]).transpose(),mu1,Sigma1)    # 2d Normal distribution 1

        F1 = np.reshape(F1,(len(x2),len(x1))).transpose()

        #Distribution function r0 <r <r_max
        mu2 = np.array([slice.shape[0]//2, slice.shape[1]//2])                        #Average
        #Sigma2 = 1/(2*math.pi)*np.array([[Xie2,Xie2*s], [Xie2*s,Xie2]])         #Covariance matrix, alternative
        Sigma2 = np.array([[Xie2,Xie2*s], [Xie2*s,Xie2]])                     # Covariance matrix
        F2 = (r2*Xie2)*2*math.pi*mvn.pdf(np.array([X1.flatten(), X2.flatten()]).transpose(),mu2,Sigma2)    #2d Normal distribution 2

        F2 = np.reshape(F2,(len(x2),len(x1))).transpose()

        #Overlay of both distributions
        #F = (F1 + F2)/2;
        
        PDF = F1
        idx = PDF <= F2
        PDF[idx] = F2[idx]
        PDF[PDF>=1] = 1

        return PDF

    def forward(self, input, target):
        w = self.weight.to(input.device)
        input = input * w
        target = target * w
        return self.loss_func(input, target)