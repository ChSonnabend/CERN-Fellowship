import torch
import torch.nn as nn
import numpy as np

def custom_mse_loss(input, target, reduction='mean'):
    if reduction=='mean':
        return torch.mean(torch.pow((input - target).flatten(),2))
    if reduction=='sum':
        return torch.sum(torch.pow((input - target).flatten(),2))
    
def weighted_mse_loss(input, target, weights=None, reduction='mean'):
    if weights is None:
        weights = 1
    if reduction=='mean':
        return torch.mean(torch.pow(weights*(input - target).flatten(),2))
    if reduction=='sum':
        return torch.sum(torch.pow(weights*(input - target).flatten(),2))
        
def hyperparamopt_loss(validation_loss, num_layers, num_neurons_per_layer, input_size, output_size):
    
    return 10**(validation_loss*250) + np.log10((input_size+1)*num_neurons_per_layer + (num_layers-1)*(num_neurons_per_layer**2 + num_neurons_per_layer) + (num_neurons_per_layer+1)*output_size)
        
def f1_loss(input, target, is_training=True):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    
    if input.ndim == 2:
        input = input.argmax(dim=1)
        
    tp = (target * input).sum().to(torch.float32)
    tn = ((1 - target) * (1 - input)).sum().to(torch.float32)
    fp = ((1 - target) * input).sum().to(torch.float32)
    fn = (target * (1 - input)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1