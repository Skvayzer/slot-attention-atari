import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

class CLEVRwithMasks(Dataset):
    def __init__(self, path_to_dataset, resize, max_objs=6, get_masks=False):
        data = np.load(path_to_dataset)
        # print("\n\n STARTED SELECTION", file=sys.stderr, flush=True)

        # raw_images = data['images']
        print("\n\n processed images", file=sys.stderr, flush=True)

        # if get_masks:
        #     # self.masks = torch.squeeze(torch.tensor(data['masks']))
        #     # raw_masks = data['masks']
        #     self.masks = np.empty((0, 11, 1, 240, 320))
        #
        # self.images = np.empty((0, 3, 240, 320))
        # print("\n\n STARTED SELECTION", file=sys.stderr, flush=True)
        #
        # for i, v in enumerate(data['visibility']):
        #     print("\n\n", i, file=sys.stderr, flush=True)
        #     print("\n\n", v, file=sys.stderr, flush=True)
        #
        #     if i > 1000:
        #         break
        #     if sum(v) > max_objs+1:
        #         continue
        #     # print("\n\nATTENTION! raw imgs : ", raw_images[i].unsqueeze(dim=0).shape, file=sys.stderr, flush=True)
        #
        #     # self.images = torch.vstack((self.images, raw_images[i].unsqueeze(dim=0)))
        #     self.images = np.vstack((self.images, data['images'][i].expand_dims(dim=0)))
        #
        #     if get_masks:
        #         # print("\n\nATTENTION! raw masks : ", raw_masks.shape, file=sys.stderr, flush=True)
        #
        #         self.masks = np.vstack((self.masks, data['masks'][i].expand_dims(dim=0)))


        self.images = data['images']
        if get_masks:
            self.masks = data['masks']
        self.visibility = data['visibility']
        self.resize = resize
        self.get_masks = get_masks
        # self.visibility = data['visibility']
        self.image_size = self.images[0].shape
        self.image_transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            # torchvision.transforms.CenterCrop((192, 192)),
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        self.mask_transform = torchvision.transforms.Compose([
            # torchvision.transforms.CenterCrop((192, 192)),
            torchvision.transforms.Resize(resize)
        ])
        print("\n\nDONE SELECTION", self.images.shape, file=sys.stderr, flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print("\n\nATTENTION! item : ", self.images[idx].shape, file=sys.stderr, flush=True)
        # print("\n\nATTENTION! item : ", self.masks[idx].shape, file=sys.stderr, flush=True)
        # print(idx, file=sys.stderr, flush=True)
        image = self.images[idx][:, 29:221, 64:256]
        image = torchvision.transforms.functional.to_pil_image(image.transpose(1, 2, 0))
        # image = torchvision.transforms.functional.crop(image, top=64, left=29, height=192, width=192)

        # image = self.image_transform(self.images[idx])
        image = self.image_transform(image)

        visibility = self.visibility[idx]
        if self.get_masks:
            # mask = torchvision.transforms.functional.to_pil_image(self.masks[idx].transpose(1, 2, 0))
            # mask = torchvision.transforms.functional.crop(mask, top=64, left=29, height=192, width=192)
            # mask = self.mask_transform(self.masks[idx])
            mask = torch.tensor(self.masks[idx][:, :, 29:221, 64:256])
            transformed_mask = torch.zeros((11, 1) + self.resize)
            # print("\n\nATTENTION! transformed_mask: ", transformed_mask.shape, self.resize, file=sys.stderr, flush=True)

            for i in range(11):
                transformed_mask[i] = self.mask_transform(mask[i])
            transformed_mask = transformed_mask.float() / 255
        # print("\n\nATTENTION! item : ", self.masks[idx].shape, file=sys.stderr, flush=True)
        #     print("\n\nATTENTION! transformed_mask1: ", transformed_mask.shape, file=sys.stderr, flush=True)

        # print("\n\nATTENTION! clevr with masks image max/min: ", torch.max(image), torch.min(image), file=sys.stderr, flush=True)
        # print("\n\nATTENTION! clevr with masks mask max/min: ", torch.max(mask), torch.min(mask), file=sys.stderr, flush=True)

        return {
            'image': image * 2 - 1,
            'mask': transformed_mask if self.get_masks else [],
            'visibility': visibility
        }