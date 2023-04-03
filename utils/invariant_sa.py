import sys

import torch
import numpy as np
from math import cos, sin

def rot(angle):
    return torch.tensor([[cos(angle), -sin(angle)],
                         [sin(angle), cos(angle)]]).cuda()
def postprocess(v1, v2):
    R = torch.stack((v1, v2), dim=1)
    print(f"\n\nATTENTION! R: {R.shape} ", file=sys.stderr, flush=True)

    det = torch.linalg.det(R)
    # for det in dets:
    idx = det < 0
    print(f"\n\nATTENTION! R[idx] : {R[idx].shape} ", file=sys.stderr, flush=True)
    print(f"\n\nATTENTION! torch.stack((v1[idx], v2[idx]), dim=0): {torch.stack((v1[idx], v2[idx]), dim=1).shape} ", file=sys.stderr, flush=True)

    R[idx] = torch.stack((v1[idx], v2[idx]), dim=1)
    # if det < 0:
    #     v1, v2 = v2, v1
    #     R =
    angle = torch.arccos(R[:, 0, 0])
    R[angle > np.pi/4] = rot(np.pi/4)
    R[angle < 0] = torch.eye(2).cuda()
    # if angle > np.pi/4:
    #     return rot(np.pi/4)
    # if angle < 0:
    #     return torch.eye(2)
    return R




