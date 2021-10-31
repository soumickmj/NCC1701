#!/usr/bin/env python

"""
This module can calculate output shape for various layers of neural network.

This code is created with the help of the following
Reference (Conv & Pool): http://cs231n.github.io/convolutional-networks
Reference (Conv Transpose): https://arxiv.org/pdf/1603.07285.pdf

Convolution output visualizer online: -
Convolution Visulizer: https://ezyang.github.io/convolution-visualizer/index.html

"""

#import torch.nn as nn

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Currently for Conv2D and 3D, ConvTranspose2D and 3D, Pool2D and 3D"

def ForConv2d(w_input, h_input, n_filters_input, n_filter_output, kernel_size, stride, padding, dilation = None):
    """Calculates output size for Convolution 2D, including wieghts per filter, total no of weights and no of biases"""

    #According to reference formula, the names of input params are as follows:
    #(w_input - W1, h_input - H1, n_filters_input - D1, n_filter_output - D2/K, kernel_size - F, stride - S, padding - P)
    if dilation is None:
        w_output = (w_input - kernel_size + 2*padding)/stride + 1
        h_output = (h_input - kernel_size + 2*padding)/stride + 1
        wieghts_per_filter = kernel_size * kernel_size * n_filters_input
        total_n_weights = wieghts_per_filter * n_filter_output
        n_biases = n_filter_output
    else:
        w_output = [w_input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1)]/stride + 1
        h_output = [h_input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1)]/stride + 1
        wieghts_per_filter = None
        total_n_weights = None
        n_biases = None
    return w_output, h_output, n_filter_output, wieghts_per_filter, total_n_weights, n_biases

def ForConv3d(w_input, h_input, d_input, n_filters_input, n_filter_output, kernel_size, stride, padding, dilation = None):
    """Calculates output size for Convolution 3D, including wieghts per filter, total no of weights and no of biases"""

    if dilation is None:
        w_output = (w_input - 5 + 2*padding)/stride + 1
        h_output = (h_input - 5 + 2*padding)/stride + 1
        d_output = (d_input - 1 + 2*padding)/stride + 1
        wieghts_per_filter = kernel_size * kernel_size * n_filters_input
        total_n_weights = wieghts_per_filter * n_filter_output
        n_biases = n_filter_output
    else:
        w_output = (w_input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1))/stride + 1
        h_output = (h_input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1))/stride + 1
        d_output = (d_input + 2*padding - kernel_size - (kernel_size-1)*(dilation-1))/stride + 1
        wieghts_per_filter = None
        total_n_weights = None
        n_biases = None
    return w_output, h_output, d_output, n_filter_output, wieghts_per_filter, total_n_weights, n_biases

def ForPool2d(w_input, h_input, n_filters_input, kernel_size, stride):
    """Calculates output size for Max Pooling 2D"""

    #According to reference formula, the names of input params are as follows:
    #(w_input - W1, h_input - H1, n_filters_input - D1, kernel_size - F, stride - S)
    w_output = (w_input - kernel_size)/stride + 1
    h_output = (h_input - kernel_size)/stride + 1
    n_filter_output = n_filters_input
    return w_output, h_output, n_filter_output

def ForPool3d(w_input, h_input, d_input, n_filters_input, kernel_size, stride):
    """Calculates output size for Max Pooling 3D"""

    w_output = (w_input - kernel_size)/stride + 1
    h_output = (h_input - kernel_size)/stride + 1
    d_output = (d_input - kernel_size)/stride + 1
    n_filter_output = n_filters_input
    return w_output, h_output, d_output, n_filter_output

def ForConvTranspose2d(w_input, h_input, n_filters_input, n_filter_output, kernel_size, stride, padding, output_padding = 0):
    """Calculates output size for Convolution Transpose 2D, including wieghts per filter, total no of weights and no of biases, though these additional infos can be wrong"""

    #According to reference formula, the names of input params are as follows:
    #(i' - W1, i' - H1, kernel_size - K, stride - S, padding - P)
    #o = (i -1)*s - 2*p + k + output_padding 
    w_output = stride * (w_input - 1) + kernel_size - 2 * padding + output_padding
    h_output = stride * (h_input - 1) + kernel_size - 2 * padding + output_padding
    wieghts_per_filter = kernel_size * kernel_size * n_filters_input
    total_n_weights = wieghts_per_filter * n_filter_output
    n_biases = n_filter_output
    return w_output, h_output, n_filter_output, wieghts_per_filter, total_n_weights, n_biases

def ForConvTranspose3d(w_input, h_input, d_input, n_filters_input, n_filter_output, kernel_size, stride, padding):
    """Calculates output size for Convolution Transpose 3D, including wieghts per filter, total no of weights and no of biases, though these additional infos can be wrong"""

    w_output = stride * (w_input - 1) + kernel_size - 2 * padding
    h_output = stride * (h_input - 1) + kernel_size - 2 * padding
    d_output = stride * (d_input - 1) + kernel_size - 2 * padding
    wieghts_per_filter = kernel_size * kernel_size * n_filters_input
    total_n_weights = wieghts_per_filter * n_filter_output
    n_biases = n_filter_output
    return w_output, h_output, d_output, n_filter_output, wieghts_per_filter, total_n_weights, n_biases

##print(ForConv2d(128, 128, 12, 12, 3,1,1))
print(ForConv2d(128, 128, 12, 12, 5, 1, 3))
#print(ForConv3d(32,26,4, 12, 12, 3, 1, 1,1))
#print(ForConv3d(61, 61, 1, 12, 12, 5, 1, 0))
#print(ForPool3d(57, 57, 1, 12, 2, 2))

#print(ForPool2d(258,258,128,3,2))

print(ForConvTranspose2d(64,64, 2048, 1024, 3, 2, 2, 2))