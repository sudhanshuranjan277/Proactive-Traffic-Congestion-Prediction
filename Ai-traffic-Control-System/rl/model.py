"""
DDQN model components for traffic signal control.
"""

import torch.nn as nn

from config import DDQN_HIDDEN_DIM


class QNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = DDQN_HIDDEN_DIM
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
