import os
import random
import sys

from torchvision.transforms import transforms

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from argparse import ArgumentParser
import argparse
from datasets import CLEVR, CLEVRTEX, CLEVR_Mirror
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance



# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--path_real", type=str)
program_parser.add_argument("--path_gen", type=str)

program_parser.add_argument("--dataset", type=str)

# Experiment parameters
program_parser.add_argument("--device", default='gpu')
program_parser.add_argument("--batch_size", type=int, default=64)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--sa_state_dict", type=str, default='./quantised_sa_rep/clevr7_od')
program_parser.add_argument("--pretrained", type=bool, default=False)
program_parser.add_argument("--beta", type=float, default=2.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--quantization", default=True, action=argparse.BooleanOptionalAction)
program_parser.add_argument("--alter", default=False, action=argparse.BooleanOptionalAction)



# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

seed_everything(args.seed, workers=True)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
dataset = args.dataset
val_dataset = None
collation = None

# class ImageDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample



real_data = ImageFolder(root=args.path_real, transform=transforms.ToTensor())
gen_data = ImageFolder(root=args.path_gen, transform=transforms.ToTensor())

fid = FrechetInceptionDistance(feature=64, normalize=True)
print("\n\nATTENTION! data: ", real_data[0][0].unsqueeze(dim=0), real_data[0][0].unsqueeze(dim=0).shape, '\n\n', file=sys.stderr, flush=True)

for i in range(len(real_data)):
    # print("\n\nATTENTION! data: ", real_data[i], '\n\n', file=sys.stderr, flush=True)
    fid.update(real_data[i][0].unsqueeze(dim=0), real=True)
    fid.update(gen_data[i][0].unsqueeze(dim=0), real=False)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("\n\nATTENTION! fid: ", fid.compute(), '\n\n', file=sys.stderr, flush=True)



