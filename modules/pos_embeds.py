import sys

import torch
from torch import nn

from utils import build_grid


class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(4, hidden_size)
        grid = torch.Tensor(build_grid(resolution))
        print(f"\n\nATTENTION! GRID: {self.grid.shape} ", file=sys.stderr, flush=True)

        self.grid = nn.Parameter(grid, requires_grad=False)

        
    def forward(self, inputs):
        pos_emb = self.linear(self.grid).moveaxis(3, 1)
        # print("\n\nATTENTION! inputs : ", inputs.shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! pos_emb : ", pos_emb.shape, file=sys.stderr, flush=True)




        return inputs + pos_emb, pos_emb.expand(inputs.shape[0], -1, -1, -1),
