"""
These functions are adapted from github.com/Officium/RL-Experiments

"""


import numpy as np
import torch
import torch.nn as nn



def scale_ob(array, device, scale):
    return torch.from_numpy(array.astype(np.float32) * scale).to(device)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)
