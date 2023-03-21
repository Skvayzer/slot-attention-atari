import sys

import numpy as np


def build_grid(resolution, double_sided=True):
    """
    :param resolution: tuple of 2 numbers
    :return: grid for positional embeddings built on input resolution
    """
    # print(f"\n\nATTENTION! BUILDING GRID ", file=sys.stderr, flush=True)

    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    # print(f"\n\nATTENTION! GRID {grid}", file=sys.stderr, flush=True)

    grid = np.stack(grid, axis=-1)
    # print(f"\n\nATTENTION! GRID {grid.shape}", file=sys.stderr, flush=True)

    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    # print(f"\n\nATTENTION! GRID {grid.shape}", file=sys.stderr, flush=True)

    grid = np.expand_dims(grid, axis=0)
    # print(f"\n\nATTENTION! GRID {grid.shape}", file=sys.stderr, flush=True)

    grid = grid.astype(np.float32)
    # print(f"\n\nATTENTION! GRID {grid.shape}", file=sys.stderr, flush=True)

    if double_sided:
        return np.concatenate([grid, 1.0 - grid], axis=-1)
    else:
        return grid
