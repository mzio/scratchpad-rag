"""
Model loss functions and objectives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from .correct import ZeroOneLoss
from .perplexity import Perplexity, BitsPerCharacter
from .soft_xent import SoftCrossEntropyLoss


def get_loss(method: str, reduction: str='none', 
             ignore_index: int=-100, **kwargs: any) -> any:
    """
    Different loss functions depending on the dataset / task
    """
    # Classification
    if method == 'cross_entropy':
        return CrossEntropyLoss(reduction=reduction,
                                ignore_index=ignore_index)
    elif method == 'zero_one':
        return ZeroOneLoss(reduction=reduction, correct=False)
    elif method == 'correct':
        return ZeroOneLoss(reduction=reduction, correct=True)
    
    # Language modeling
    elif method == 'perplexity':
        return Perplexity(ignore_index=ignore_index)
    elif method == 'bpc':
        return BitsPerCharacter(ignore_index=ignore_index)
    
    # Attention distillation
    elif method == 'soft_xent':
        return SoftCrossEntropyLoss(reduction=reduction)
    
    else:
        raise NotImplementedError(f'{loss} loss not implemented!')
        
        
def init_loss(**kwargs) -> any:
    """Same thing as get_loss"""
    return get_loss(**kwargs)
    