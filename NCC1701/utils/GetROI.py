#!/usr/bin/env python

"""
This module helps to clear Non-ROI from Undersampled or Recon Image using Fully sampled as reference
to help in better accuracy calculation
Currently this module uses my matlab code, so it requires matlab engine

"""

#import matlab.engine
import numpy as np

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Partially Finished"

def ClearNonROI(Source,Target):
    """
    This function is specially written for comparing original and
    undersampled image for check quality, for clearing out the non-ROI 
    Input:-
    Fully: Fully Sampled Image
    Under: Undersampled Image of the same Fully Sampled Image
    Output:-
    Out: Undersampled Image with non-ROI area cleared, i.e. all pixels of the non-ROI area set to 0 (Black)
    """
    
    #Callling Matlab Function
    #eng = matlab.engine.start_matlab()
    #Out = eng.ClearNonROI(Source.tolist(),Target.tolist())
    #Out = np.asarray(Out)
    Out = Target
    
    return Out   