from .clevr import CLEVR
from .clevrtex import  CLEVRTEX, collate_fn
from .multi_dsprites import MultiDSprites
from .tetrominoes import Tetrominoes
from .clevrmirror import CLEVR_Mirror
from .celeba import CelebA
from .clevr_with_masks import CLEVRwithMasks

__all__ = ['CLEVR', 'CLEVRTEX', 'CLEVR_Mirror', 'collate_fn', 'Tetrominoes', 'MultiDSprites', 'CelebA', 'CLEVRwithMasks']
