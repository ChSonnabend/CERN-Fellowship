import torch
import torch.nn as nn

class adaptiveAvgPool2d_layer(nn.Module):
    
    def __init__(self, params, activation, weight_init, verbose=False, device=None, dtype=torch.float, **options):
        
        super().__init__()
        
        self.outputsize = params[0]
        self.avg = nn.AdaptiveAvgPool2d(self.outputsize)
        
        print("Adaptive average pooling layer, Output size: ", params)
        
    def forward(self, X):
        
        return self.avg(X)