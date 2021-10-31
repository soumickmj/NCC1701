"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.measure import compare_psnr, compare_ssim


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred, return_map=False):
    """ Compute Structural Similarity Index Metric (SSIM). """
    if len(gt.shape) == 3: ##Added by Soumick
        return compare_ssim(
            gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max(), full=return_map
        )
    else:
        return compare_ssim(
            gt, pred, multichannel=False, data_range=gt.max(), full=return_map
        )

##added by Soumick
def error_percent(gt, pred):
    """ Compute the precent of Error """
    return np.mean(pred != gt) * 100

def ssd(gt, pred):
    return np.sum( (gt - pred) ** 2 )
##add by Soumick zone ends

###Commented out by Soumick
#from runstats import Statistics
#METRIC_FUNCS = dict(
#    MSE=mse,
#    NMSE=nmse,
#    PSNR=psnr,
#    SSIM=ssim,
#)


#class Metrics:
#    """
#    Maintains running statistics for a given collection of metrics.
#    """

#    def __init__(self, metric_funcs):
#        self.metrics = {
#            metric: Statistics() for metric in metric_funcs
#        }

#    def push(self, target, recons):
#        for metric, func in METRIC_FUNCS.items():
#            self.metrics[metric].push(func(target, recons))

#    def means(self):
#        return {
#            metric: stat.mean() for metric, stat in self.metrics.items()
#        }

#    def stddevs(self):
#        return {
#            metric: stat.stddev() for metric, stat in self.metrics.items()
#        }

#    def __repr__(self):
#        means = self.means()
#        stddevs = self.stddevs()
#        metric_names = sorted(list(means))
#        return ' '.join(
#            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
#        )


#def evaluate(args, recons_key):
#    metrics = Metrics(METRIC_FUNCS)

#    for tgt_file in args.target_path.iterdir():
#        with h5py.File(tgt_file) as target, h5py.File(
#          args.predictions_path / tgt_file.name) as recons:
#            if args.acquisition and args.acquisition != target.attrs['acquisition']:
#                continue
#            target = target[recons_key].value
#            recons = recons['reconstruction'].value
#            metrics.push(target, recons)
#    return metrics
