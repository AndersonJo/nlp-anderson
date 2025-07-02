import torch
from torch import nn

class Expert(nn.Module):
    """
    A simple feed-forward network, which will be used as an expert in the MoE layer.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
