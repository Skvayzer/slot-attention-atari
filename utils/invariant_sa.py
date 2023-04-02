import sys

import torch
import numpy as np
from math import cos, sin

def rot(angle):
    return torch.tensor([[cos(angle), -sin(angle)],
                         [sin(angle), cos(angle)]])
def postprocess(v1, v2):
    R = torch.stack((v1, v2), dim=1)
    print(f"\n\nATTENTION! R: {R.shape} ", file=sys.stderr, flush=True)

    det = torch.linalg.det(R)
    # for det in dets:
    if det < 0:
        v1, v2 = v2, v1
        R = torch.stack((v1, v2), dim=0)
    angle = torch.arccos(R[0, 0])
    if angle > np.pi/4:
        return rot(np.pi/4)
    if angle < 0:
        return torch.eye(2)
    return R




