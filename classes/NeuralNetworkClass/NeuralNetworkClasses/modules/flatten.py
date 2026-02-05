import torch
import torch.nn as nn

class flatten(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, device=None, dtype=torch.float, **options):
        
        super().__init__()
            
        self.act = activation
        self.flatten = torch.flatten
        
        if verbose:
            print("Flattening layer")
            
    def forward(self, X):
        
        return self.act(self.flatten(X, 1))
    