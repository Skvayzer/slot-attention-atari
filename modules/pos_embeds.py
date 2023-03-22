import sys

import torch
from torch import nn

from utils import build_grid


class ISAPosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(2, hidden_size)

        grid = torch.Tensor(build_grid(resolution, double_sided=False))

        self.grid = nn.Parameter(grid, requires_grad=False)

        
    def forward(self, inputs, grid=None):
        if grid is None:
            grid = self.grid
        print("\n\nATTENTION! inputs : ", inputs.shape, file=sys.stderr, flush=True)

        pos_emb = self.linear(grid).moveaxis(3, 1)
        print("\n\nATTENTION! pos_emb : ", pos_emb.shape, file=sys.stderr, flush=True)

        return inputs + pos_emb, pos_emb.expand(inputs.shape[0], -1, -1, -1),


class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution, proj_dim=4):
        super().__init__()
        self.linear = nn.Linear(proj_dim, hidden_size)

        grid = torch.Tensor(build_grid(resolution))

        self.grid = nn.Parameter(grid, requires_grad=False)

    def forward(self, inputs, grid=None):
        if grid is None:
            grid = self.grid
        pos_emb = self.linear(grid).moveaxis(3, 1)
        print("\n\nATTENTION! inputs : ", inputs.shape, file=sys.stderr, flush=True)
        print("\n\nATTENTION! pos_emb : ", pos_emb.shape, file=sys.stderr, flush=True)

        return inputs + pos_emb, pos_emb.expand(inputs.shape[0], -1, -1, -1),

