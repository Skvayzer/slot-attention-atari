import os
import sys


sys.path.append("..")

from pytorch_lightning.loggers import WandbLogger

from argparse import ArgumentParser
import argparse

from models import SlotAttentionAE
from torch.optim import lr_scheduler

from torchvision.utils import save_image

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

from random import randrange

import gym
from ale_py.roms import Seaquest

from ale_py import ALEInterface
ale = ALEInterface()
ale.loadROM(Seaquest)

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
program_parser.add_argument("--beta", type=float, default=1.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--quantization", default=False, action=argparse.BooleanOptionalAction)




# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# h, w
resize = (128, 128)

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.images = collections.deque(maxlen=len)
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)


    def update(self, img=None, state=None, action=None, reward=None, done=None):

        self.images.append(img)
        if state!=None:
            # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
            # and actions which leads to a mismatch when we sample from memory.
            if not done:
                self.state.append(state)

            self.action.append(action)
            self.rewards.append(reward)
            self.is_done.append(done)

    def __len__(self):
        return len(self.images)

    def preprocess(self):
        print("Preprocessing", file=sys.stderr, flush=True)
        images = np.array(self.images) / 255
        images = images * 2 - 1
        # images = images.permute(0, 3, 2, 1)
        # images = torchvision.transforms.CenterCrop((160, 160))(images)
        # images = F.resize(images, resize).permute(0, 1, 3, 2)
        print("batch shape", images.shape, file=sys.stderr, flush=True)
        self.images = images

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.images)
        idx = random.sample(range(0, n-1), batch_size)
        images = self.images[idx]

        return images
        # return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
        #        torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
        #        torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.images.clear()
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def train_step(batch_size, model, optimizer, memory):
    model.train()
    print("Batch", file=sys.stderr, flush=True)

    # states, actions, next_states, rewards, is_done = memory.sample(batch_size)
    images = memory.sample(batch_size)

    loss = model.training_step(images, optimizer)
    return loss


# def evaluate_step(model, env, repeats=8):
#     """
#     Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
#     episode reward.
#     """
#     model.eval()
#     images = torch.empty((16, 3, 128, 128))
#
#     for _ in range(repeats):
#         state = env.reset()
#
#         img = env.render()
#         done = False
#         # while not done:
#         playtime = randrange(128)
#         i = 0
#         episode_imgs = []
#         while i < playtime and not done:
#
#             img = torch.Tensor(img).to(device).float() / 255
#             img = img * 2 - 1
#             print("img shape", img.shape, file=sys.stderr, flush=True)
#             img = img.permute(2, 0, 1)
#             print("img ", img, torch.max(img), torch.min(img), file=sys.stderr, flush=True)
#
#
#
#             img = F.resize(img, resize) #.permute(0, 2, 1)
#             print("img shape", img.shape, file=sys.stderr, flush=True)
#             images[i] = img
#             action = env.action_space.sample()
#             state, reward, done, _, _ = env.step(action)
#             img = env.render()
#
#
#     with torch.no_grad():
#         model.validation_step(images)
#
#     # wandb.log({
#     #     'orig images': [wandb.Image(img) for img in images],
#     # })
#     model.train()

def evaluate_step(model, env, val_memory):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    model.eval()

    images = val_memory.sample(64)

    with torch.no_grad():
        model.validation_step(images)

    model.train()


def generate_memory(env, model, episodes=20, max_memory_size=20000, mode='train'):
    # memory = Memory(max_memory_size)
    i = 0
    for e in range(episodes):
        print(f"Memory Episode {e}", file=sys.stderr, flush=True)

        state = env.reset()

        # img = env.render()
        done = False
        # while not done:
        while not done:
            i += 1
            # memory.update(state)
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if i % 4 == 0:
                continue
            # print("state shape", state.shape, file=sys.stderr, flush=True)
            state = torch.tensor(state).permute(2, 0, 1) / 255
            save_image(state, os.path.join("/mnt/data/users_data/smirnov/sa_atari/datasets/seaquest", mode, mode + '_' + str(i) + '.png'))
            # img = env.render()

            print(i, file=sys.stderr, flush=True)

            if i > max_memory_size:
                return

    return #memory

def collate_fn(batch):
    print(batch, file=sys.stderr, flush=True)

    images = torch.stack([b[0] for b in batch])

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
dataset = args.dataset

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

train_dataset = ImageFolder(root=args.train_path, transform=transforms)
val_dataset = ImageFolder(root=args.val_path, transform=transforms)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                          drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        drop_last=True, collate_fn=collate_fn)

monitor = 'Validation MSE'
autoencoder = SlotAttentionAE(**dict_args, resolution=resize)
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
                     max_epochs=args.max_epochs,
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

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)
# Test
trainer.test(dataloaders=val_loader, ckpt_path=None)
wandb.finish()



