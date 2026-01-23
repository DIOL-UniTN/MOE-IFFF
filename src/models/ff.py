import torch
import math
from torch import nn
from typing import Optional, Tuple

class FF(nn.Module):
    def __init__(self, in_features: int, width: int, out_features: int):
        super().__init__()
        self.name = "FF"
        self.in_features = in_features
        self.width = width
        self.out_features = out_features
        self.activation = nn.ReLU()

        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(width, out_features)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

    def cal_complexity(self) -> Tuple[float, float]: # Return #params, MAC
        n_params = (self.in_features * self.width + self.width * self.out_features)
        macs = n_params
        return n_params, macs

    def get_config(self):
        n_params, macs = self.cal_complexity()
        return {
                'in_features': self.in_features,
                'width': self.width,
                'out_features': self.out_features,
                'macs': macs,
                'n_params': n_params,
                'model_name': self.name,
                }
