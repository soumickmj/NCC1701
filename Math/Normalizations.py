#!/usr/bin/env python

"""
This model deals in various normalization methods for various data

Jobs:-
    1. Normalize Fourier Space
    2. Normalize Hartley Space
    3. Min-Max Normalization
"""

import numpy as np   

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished, but more might be added later"


#Normalize Fourier Space
def fnorm(x):
    return x/abs(x).max()

#Normalize Hartley Space, directly in hartley space. For using kspace, use norm_with_fnorm parameter of the hartley transform
def hnorm(x):
    return x/abs(x).max()

