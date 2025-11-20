import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 512),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)