import numbers
import torch.nn as nn
import torch


class NormLayer(nn.Module):
    __constants__ = ['norm_shape', 'weight', 'bias', 'eps']

    def __init__(self, norm_shape, eps=1e-6):
        super(NormLayer, self).__init__()
        if isinstance(norm_shape, numbers.Integral):
            norm_shape = (norm_shape,)
        self.norm_shape = norm_shape

        # create two trainable parameters to do affine tuning
        self.weight = nn.Parameter(torch.ones(*self.norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(*self.norm_shape), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.weight * (x - x.mean(dim=-1, keepdim=True))
        norm /= (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
