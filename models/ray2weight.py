import torch
from torch import nn


class Ray2Alpha(nn.Module):
    def __init__(self, window=11, init_bias=False):
        super().__init__()
        self.mlp1 = nn.Linear(window*2, 256)
        self.mlp2 = nn.Linear(256, 256)
        self.mlp3 = nn.Linear(256, 256-window*2)
        self.mlp4 = nn.Linear(256, 256)
        self.mlp5 = nn.Linear(256, 256)
        self.mlp6 = nn.Linear(256, 1)
        if init_bias:
            nn.init.constant_(self.mlp6.bias, -0.2)
        self.activ = torch.nn.Softplus(beta=100)
        self.sharpness = 1.0

    def forward(self, sdf, sharpness=None):
        # sdf: [N, window, 2]
        sdf = sdf.reshape(sdf.shape[0], -1)     # [N, window+window]
        x = self.mlp1(sdf)
        x = self.activ(x)
        x = self.mlp2(x)
        x = self.activ(x)
        x = self.mlp3(x)
        x = self.activ(x)
        x = torch.cat([x, sdf], -1)
        x = self.mlp4(x)
        x = self.activ(x)
        x = self.mlp5(x)
        x = self.activ(x)
        x = self.mlp6(x)
        if sharpness is not None:
            x = torch.sigmoid(x * sharpness)
        else:
            x = torch.sigmoid(x * self.sharpness)
        return x
