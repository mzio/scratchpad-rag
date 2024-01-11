import torch
import torch.nn as nn


class ZeroOneLoss(nn.Module):
    """
    By default will return the number of incorrect samples
    """
    def __init__(self, reduction: str = 'mean', 
                 correct: bool = False, 
                 ignore_index: int = -100):
        super().__init__()
        self.reduction = reduction
        self.correct   = correct
        self.ignore_index = ignore_index
        
    def forward(self, y_pred, y_true):
        _, y_pred = torch.max(y_pred.data, -1)
        loss = (y_pred == y_true).float()
        if not self.correct:
            loss = 1. - loss
        return self.reduce(loss)
        
    def reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
