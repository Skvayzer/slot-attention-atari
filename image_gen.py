import os
import random
import sys

import torchvision.transforms

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from argparse import ArgumentParser
import argparse
from datasets import CLEVR, CLEVRTEX, CLEVR_Mirror
from torchvision.datasets import CelebA
from models import SlotAttentionAE
import wandb
from datasets import collate_fn
from datasets import MultiDSprites, CLEVRwithMasks
from utils import adjusted_rand_index


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
program_parser.add_argument("--train_path", type=str)
program_parser.add_argument("--val_path", type=str)

program_parser.add_argument("--dataset", type=str)

# Experiment parameters
program_parser.add_argument("--device", default='gpu')
program_parser.add_argument("--batch_size", type=int, default=64)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--sa_state_dict", type=str, default='./quantised_sa_rep/clevr7_od')
program_parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
program_parser.add_argument("--beta", type=float, default=2.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--quantization", default=True, action=argparse.BooleanOptionalAction)
program_parser.add_argument("--alter", default=False, action=argparse.BooleanOptionalAction)
program_parser.add_argument("--num_batches", type=int, default=5)
program_parser.add_argument("--save_dir", type=str, default='./generated_images/clevr/vq-sa')




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
# Logger
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
dataset = args.dataset
val_dataset = None
collation = None


if dataset == 'clevr':
    #max 6 objects
    if args.val_path != None:
        val_dataset = CLEVRwithMasks(os.path.join(args.val_path, 'clevr_with_masks_val.npz'), resize=(128, 128))
    else:
        val_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'val'),
                            scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_val_scenes.json'),
                            max_objs=6)
elif dataset == 'clevr-mirror':
    clevr_mirror = CLEVR_Mirror(images_path=os.path.join(args.train_path, 'images'),
                      scenes_path=os.path.join(args.train_path, 'scenes'),
                      max_objs=6)

    test_size = int(0.2 * len(clevr_mirror))
    train_size = len(clevr_mirror) - test_size
    val_dataset = Subset(clevr_mirror, list(range(test_size)))


elif dataset == 'clevr-tex':
    val_dataset = CLEVRTEX(
        args.train_path, # Untar'ed
        dataset_variant='full', # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
        split='val',
        max_obj=6,
        crop=True,
        resize=(128, 128),
        return_metadata=True # Useful only for evaluation, wastes time on I/O otherwise
    )
    collation = collate_fn
elif dataset == 'celeba':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])
    print("\n\nATTENTION! celeba path: ", args.train_path, '\n\n', file=sys.stderr, flush=True)

    val_dataset = CelebA(root=args.train_path, split='valid', target_type='attr', transform=transforms, download=True)

elif dataset == 'tetrominoes':
    val_dataset = MultiDSprites(path_to_dataset=(args.train_path + '/tetrominoes_val.npz'), mode='tetraminoes')

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        drop_last=True, collate_fn=collation)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("\n\nATTENTION! quantize: ", args.quantization, '\n\n', file=sys.stderr, flush=True)

# model
dict_args = vars(args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dataset == 'tetrominoes':
    autoencoder = SlotAttentionAE(resolution=(35, 35), hidden_size = 32, decoder_initial_size=(35, 35),
                     num_slots=4, **dict_args)
else:
    autoencoder = SlotAttentionAE(**dict_args)
autoencoder = autoencoder.to(device)
project_name = 'object_detection_' + dataset



monitor = 'Validation MSE'


# print("\n\nATTENTION! ckpt: ", checkpoint, file=sys.stderr, flush=True)


if len(args.from_checkpoint) > 0:
    checkpoint = torch.load(args.from_checkpoint)['state_dict']
    autoencoder.load_state_dict(state_dict=checkpoint, strict=False)
elif args.pretrained:
    state_dict = torch.load(args.sa_state_dict)
    autoencoder.load_state_dict(state_dict=state_dict, strict=False)


num_slots = 7
count = 0
for i in range(args.num_batches):
    batch = next(iter(val_loader))
    imgs = batch['image'].to(device)

    result, recons, _, pred_masks = autoencoder(imgs)

    for j, img in enumerate(imgs):
        filepath = os.path.join(args.save_dir, f"{args.task}_{count}.png")
        torchvision.utils.save_image((img/2 + 0.5), filepath)
        count+=1


