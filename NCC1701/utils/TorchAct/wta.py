#!/usr/bin/env python

"""
Winnter take it all as an activation function

"""

import torch
import math

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class KWinnersTakeAll_Grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        original_shape = tensor.shape
        tensor = tensor.flatten(start_dim = 2)
        batch_size, _, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=2, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[:,: , active_indices] = 1
        tensor[~mask_active] = 0
        ctx.save_for_backward(mask_active)
        tensor = tensor.view(original_shape)
        return tensor

class KWinnersTakeAllNew(torch.nn.Module):

    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = torch.tensor([sparsity], requires_grad=True)
    
    def forward(self, tensor):
        original_shape = tensor.shape
        tensor = tensor.flatten(start_dim = 2)
        batch_size, feature_no , embedding_size = tensor.shape
        k_active = torch.ceil(self.sparsity * embedding_size)
        _, active_indices = torch.topk(tensor, k_active.item, -1)
        mask_active = torch.zeros(tensor.shape, device=tensor.device)
        #mask_active[active_indices] = 1
        ta_b = torch.arange(batch_size).unsqueeze(1).unsqueeze(1)
        ta_f = torch.arange(feature_no).unsqueeze(1).unsqueeze(1).permute([1,0,2])
        mask_active[ta_b, ta_f, active_indices] = 1
        tensor = tensor*mask_active
        tensor = tensor.reshape(original_shape)
        return tensor

class KWinnersTakeAllLearnt(torch.nn.Module):

    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity
    
    def forward(self, tensor):
        original_shape = tensor.shape
        tensor = tensor.flatten(start_dim = 2)
        batch_size, feature_no , embedding_size = tensor.shape
        k_active = torch.ceil(self.sparsity * embedding_size).int()
        _, active_indices = torch.topk(tensor, k_active, -1)
        mask_active = torch.zeros(tensor.shape, device=tensor.device)
        #mask_active[active_indices] = 1
        ta_b = torch.arange(batch_size).unsqueeze(1).unsqueeze(1)
        ta_f = torch.arange(feature_no).unsqueeze(1).unsqueeze(1).permute([1,0,2])
        mask_active[ta_b, ta_f, active_indices] = 1
        tensor = tensor*mask_active
        tensor = tensor.reshape(original_shape)
        return tensor

class KWinnersTakeAll(torch.nn.Module):

    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity
    
    def forward(self, tensor):
        original_shape = tensor.shape
        tensor = tensor.flatten(start_dim = 2)
        batch_size, feature_no , embedding_size = tensor.shape
        k_active = math.ceil(self.sparsity * embedding_size)
        _, active_indices = torch.topk(tensor, k_active, -1)
        mask_active = torch.zeros(tensor.shape, device=tensor.device)
        #mask_active[active_indices] = 1
        ta_b = torch.arange(batch_size).unsqueeze(1).unsqueeze(1)
        ta_f = torch.arange(feature_no).unsqueeze(1).unsqueeze(1).permute([1,0,2])
        mask_active[ta_b, ta_f, active_indices] = 1
        tensor = tensor*mask_active
        tensor = tensor.reshape(original_shape)
        return tensor

class realKWinnersTakeAll(torch.nn.Module):

    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity
    
    def forward(self, tensor):
        original_shape = tensor.shape
        tensor = tensor.flatten(start_dim = 2)
        batch_size, feature_no , embedding_size = tensor.shape
        k_active = math.ceil(self.sparsity)
        _, active_indices = torch.topk(tensor, k_active, -1)
        mask_active = torch.zeros(tensor.shape, device=tensor.device)
        #mask_active[active_indices] = 1
        ta_b = torch.arange(batch_size).unsqueeze(1).unsqueeze(1)
        ta_f = torch.arange(feature_no).unsqueeze(1).unsqueeze(1).permute([1,0,2])
        mask_active[ta_b, ta_f, active_indices] = 1
        tensor = tensor*mask_active
        tensor = tensor.reshape(original_shape)
        return tensor