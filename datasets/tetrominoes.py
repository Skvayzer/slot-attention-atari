from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Tetrominoes(Dataset):
    def __init__(self, path_to_dataset: Path):
        data = np.load(path_to_dataset)
        self.masks = data['masks']
        self.images = data['images']
        self.visibility = data['visibility']
        self.image_size = self.images[0].shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).float() / 255
        mask = self.masks[idx]
        visibility = self.visibility[idx]
        item = {'image': image, 'mask': mask, 'visibility': visibility}

        return item

# def tetrominoes_collate_fn(batch):
#     images = torch.stack([b['image'] for b in batch])
#     # print("TRUE MASK SHAPE: ", batch[0]['mask'].shape, file=sys.stderr, flush=True)
#     masks = torch.nn.utils.rnn.pad_sequence([b['mask'] for b in batch], batch_first=True)
#     # print("MASK POSTPROCESS SHAPE: ", masks.shape, file=sys.stderr, flush=True)
#
#     targets = []#torch.stack([b['target'] for b in batch])
#     indexes = []#torch.stack([b['index'] for b in batch])
#
#     return {
#         'image': images,
#         'mask': masks,
#         'target': targets,
#         'index': indexes
#     }