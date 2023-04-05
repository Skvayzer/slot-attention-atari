from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiDSprites(Dataset):
    def __init__(self, path_to_dataset: Path, mode=None):
        self.mode = mode
        if mode == 'clevr':
            self.masks = np.load(path_to_dataset + '_masks.npz')
            self.images = np.load(path_to_dataset + '_images.npz')
            self.visibility = np.load(path_to_dataset + '_visibility.npz')
        else:
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
        # if self.mode == 'tetrominoes':
        #     image = image * 2 - 1
        mask = self.masks[idx]
        visibility = self.visibility[idx]
        item = {'image': image, 'mask': mask, 'visibility': visibility}

        return item
