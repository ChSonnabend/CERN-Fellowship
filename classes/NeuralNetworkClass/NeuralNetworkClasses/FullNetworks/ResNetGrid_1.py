import torch
import torch.nn as nn
from RBF import gaussian_layer

class Block1(nn.Module):
    
    def __init__(self, size_in, size_out):
        
        super().__init__()
        
        self.fn1 = nn.Linear(size_in,size_in)
        self.fn2 = nn.Linear(size_in,size_in)
        self.bn1 = nn.BatchNorm1d(size_in)
        self.rbf1 = gaussian_layer([size_in,size_in], nn.Identity(),weight_init=torch.nn.init.xavier_uniform_)
        self.fn3 = nn.Linear(size_in,size_in)
        self.fn4 = nn.Linear(size_in,size_out)

    def forward(self, X):

        out = nn.Tanh()(self.fn1(X))
        out = nn.Tanh()(self.fn2(out))
        out = self.bn1(out)
        out = self.rbf1(out)
        out = nn.Tanh()(self.fn3(out))
        out = out + X
        out = self.fn4(out)

        return out


class ResNetGrid_1(nn.Module):

    def __init__(self, block, sizes):
        super().__init__()

        self.layers = nn.ModuleList()

        for counter in sizes:
            self.layers.append(block(counter[0],counter[1],counter[2]))

    def forward(self, X):

        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out
