"""
Soft-label Cross-entropy loss
"""
import torch

import torch
import torch.nn


import torch
import torch.nn as nn


class SoftCrossEntropyLoss(nn.Module):
    """
    By default will return the number of incorrect samples
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-12):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                attention_mask: torch.Tensor=None):
        y_pred = torch.clamp(y_pred, min=self.eps, max=1 - self.eps)
        y_true = torch.clamp(y_true, min=self.eps, max=1 - self.eps)
        if attention_mask is not None:
            y_true = y_true.masked_fill(~attention_mask, 0.)
        xent = -(y_true * torch.log(y_pred)).sum(dim=-1)
        return self.reduce(xent)
        
    def reduce(self, loss: torch.Tensor):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
