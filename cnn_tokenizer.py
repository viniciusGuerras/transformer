import torch
from torch import nn 
import numpy as np

class CnnTokenizer(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.convo = nn.Conv1d(block_size, block_size, block_size//2)

    def forward(self, x):
        return x + self.convo(x)