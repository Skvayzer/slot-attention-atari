import os
import sys


sys.path.append("..")

import tensorflow as tf

from argparse import ArgumentParser
import argparse

from models import SlotAttentionAE, InvariantSlotAttentionAE
from torch.optim import lr_scheduler

import torch
import numpy as np
from torch import nn
import random
import torchvision
import torchvision.transforms.functional as F
import collections
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datasets import MultiDSprites, Waymo

from random import randrange

# import gym
# from ale_py.roms import Seaquest
#
# from ale_py import ALEInterface
# ale = ALEInterface()
# ale.loadROM(Seaquest)

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
program_parser.add_argument("--beta", type=float, default=0.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--invariance", default=True, action=argparse.BooleanOptionalAction)




# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# h, w
resize = (128, 128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = args.dataset
torch.autograd.set_detect_anomaly(True)

def collate_fn(batch):
    # print(batch, "\n\n aaaaa", file=sys.stderr, flush=True)
    if dataset == 'seaquest':
        images = torch.stack([b[0] for b in batch])
    else:
        images = torch.stack([b['image'] for b in batch])

    return {
        'image': images,
    }

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
wandb.login(key='c84312b58e94070d15277f8a5d58bb72e57be7fd')

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)


project_name = 'object_detection_' + dataset

# wandb_logger = WandbLogger(project=project_name, name=f'{args.task}: nums {args.nums!r} s {args.seed} kl {args.beta}',
#                            log_model=True)
wandb.init(project=project_name)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()
])
train_dataset, val_dataset = None, None
collation = None
if dataset=='seaquest':
    train_dataset = ImageFolder(root=args.train_path, transform=transforms)
    val_dataset = ImageFolder(root=args.val_path, transform=transforms)
    collation = collate_fn
elif dataset=='tetrominoes':
    train_dataset = MultiDSprites(path_to_dataset=(args.train_path + '/tetrominoes_train.npz'), mode='tetrominoes')
    val_dataset = MultiDSprites(path_to_dataset=(args.train_path + '/tetrominoes_val.npz'), mode='tetrominoes')
elif dataset=='waymo':
    train_dataset = Waymo(path=args.train_path, train=True)
    val_dataset = Waymo(path=args.val_path, train=False)
    print(f"\n\nATTENTION! Loaded waym", file=sys.stderr, flush=True)

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
#                           drop_last=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
#                         drop_last=True, collate_fn=collation)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

print(f"\n\nATTENTION! Loaded loaders", file=sys.stderr, flush=True)

monitor = 'Validation MSE'
if dataset == 'tetrominoes':
    # autoencoder = InvariantSlotAttentionAE(**dict_args, resolution=resize, train_dataloader=train_loader)
    autoencoder = InvariantSlotAttentionAE(resolution=(35, 35), hidden_size=32, decoder_initial_size=(35, 35),
                                           enc_hidden_size=64, train_dataloader=train_loader, num_slots=4,
                                            val_num_slots=4, **dict_args)
    # autoencoder = SlotAttentionAE(resolution=(35, 35), hidden_size=32, decoder_initial_size=(35, 35),
    #                                 train_dataloader=train_loader, num_slots=4,
    #                               val_num_slots=4, **dict_args)
elif dataset=='seaquest':
    autoencoder = InvariantSlotAttentionAE(**dict_args, resolution=resize, train_dataloader=train_loader, num_slots=15,
                                            val_num_slots=15)
elif dataset == 'waymo':
    autoencoder = InvariantSlotAttentionAE(**dict_args, resolution=resize, train_dataloader=train_loader, num_slots=10,
                                           val_num_slots=10, lr=2e-4)
autoencoder.to(device)

wandb_logger = WandbLogger(project=project_name, name=f'{args.task}: nums {args.nums!r} s {args.seed} kl {args.beta}',
                           log_model=True)
# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------


# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)
every_epoch_callback = ModelCheckpoint(every_n_epochs=10, monitor=monitor)
# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# logger_callback = SlotAttentionLogger(val_samples=next(iter(val_loader)))

callbacks = [
    checkpoint_callback,
    # logger_callback,
    every_epoch_callback,
    # swa,
    # early_stop_callback,
    lr_monitor,
]
if args.pretrained:
    state_dict = torch.load(args.sa_state_dict)['state_dict']
    # state_dict = torch.load(args.sa_state_dict)
    autoencoder.load_state_dict(state_dict=state_dict, strict=False)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parameters
profiler = None  #   'simple'/'advanced'/None
accelerator = args.device
# devices = [int(args.devices)]
gpus = [0]

print(torch.cuda.device_count(), flush=True)

# trainer
trainer = pl.Trainer(accelerator=accelerator,
                     devices=[0],
                     max_steps=args.max_steps,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     )
#  precision=16,
# deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None
# else:
#     ckpt = torch.load(args.from_checkpoint)
#
#     autoencoder.load_state_dict(state_dict=ckpt, strict=False)

print(f"\n\nATTENTION! Starting Training", file=sys.stderr, flush=True)

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)
# Test
trainer.test(dataloaders=val_loader)
wandb.finish()



