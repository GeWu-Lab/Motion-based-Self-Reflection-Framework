import torch
import torch.nn as nn

import utils


class StateEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, extra_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim
        layers = [nn.Linear(self.extra_dim, self.hidden_dim), nn.ReLU()]
        if num_layers > 0:
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        
        self.net = nn.Sequential(*layers)

        self.apply(utils.weight_init)
    
    def forward(self, x):
        x = self.net(x)
        return x
