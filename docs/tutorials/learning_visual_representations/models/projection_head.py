import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """
    Projection head as defined in 
    'A Simple Framework for Contrastive Learning of Visual Representations, Chen et al. (2020)'.
    """
    def __init__(self, d_in, d_model=128, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x