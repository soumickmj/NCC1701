import torch
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import torchvision.utils as vutils
import torchcomplex.nn.functional as cF
from torchcomplex.utils.signaltools import resample

class Interpolator():
    def __init__(self, mode=None):
        if mode in ["sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]:
            self.mode = mode
        else:
            self.mode = None

    def perform_sinc(self, images, out_shape):
        axes = np.argwhere(np.equal(images.shape[2:], out_shape) == False).squeeze(1) #2 dims for batch and channel
        out_shape = [out_shape[i] for i in axes]
        return resample(images, out_shape, axis=axes+2) #2 dims for batch and channel

    def __call__(self, images, out_shape):
        if self.mode is None:
            return images
        elif images.is_complex():
            return cF.interpolate(images, size=out_shape, mode=self.mode)
        elif self.mode == "sinc":
            return self.perform_sinc(images, out_shape)
        else:
            return F.interpolate(images, size=out_shape, mode=self.mode)

def tensorboard_images(writer, inputs, outputs, targets, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, 0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, 0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                        vutils.make_grid(targets[0, 0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
