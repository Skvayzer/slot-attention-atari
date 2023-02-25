
import sys


sys.path.append("..")

from pytorch_lightning.loggers import WandbLogger

from argparse import ArgumentParser
import argparse

from models import SlotAttentionAE
from torch.optim import lr_scheduler

import torch
import numpy as np
from torch import nn
import random
import torchvision
import torchvision.transforms.functional as F
import collections
import wandb



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
program_parser.add_argument("--episodes", type=int, default=1e6)

program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--sa_state_dict", type=str, default='')
program_parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
program_parser.add_argument("--beta", type=float, default=1.)
program_parser.add_argument("--num_workers", type=int, default=4)
program_parser.add_argument("--task", type=str, default='')
program_parser.add_argument("--quantization", default=False, action=argparse.BooleanOptionalAction)


# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

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

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.images)
        idx = random.sample(range(0, n-1), batch_size)
        images = torch.Tensor(self.images)[idx].to(device).float() / 255
        images = images*2 - 1
        print("batch shape", images.shape, file=sys.stderr, flush=True)
        images = images.permute(0, 3, 2, 1)
        images = F.resize(images, resize).permute(0, 1, 3, 2)
        print("batch shape", images.shape, file=sys.stderr, flush=True)



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


def train_step(batch_size, model, optimizer, scheduler, memory):
    model.train()
    print("Batch", file=sys.stderr, flush=True)

    # states, actions, next_states, rewards, is_done = memory.sample(batch_size)
    images = memory.sample(batch_size)

    loss = model.training_step(images, optimizer, scheduler)
    return loss


def evaluate_step(model, env, repeats=8):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    model.eval()
    for _ in range(repeats):
        state = env.reset()
        img = env.render()
        done = False
        # while not done:
        for i in range(2):
            img = torch.Tensor(img).to(device).float() / 255
            img = img * 2 - 1
            print("img shape", img.shape, file=sys.stderr, flush=True)
            img = img.permute(2, 0, 1)
            print("img ", img, torch.max(img), torch.min(img), file=sys.stderr, flush=True)

            wandb.log({
                'orig images': [wandb.Image(img)],
            })
            img = torchvision.transforms.CenterCrop((160, 160))(img)
            img = F.resize(img, resize) #.permute(0, 2, 1)
            print("img shape", img.shape, file=sys.stderr, flush=True)

            with torch.no_grad():
                model.validation_step(img.unsqueeze(dim=0))
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            img = env.render()
    model.train()


def train_loop(min_episodes=20, update_step=2, batch_size=64, update_repeats=50, render_step=5,
         num_episodes=3000, seed=42, max_memory_size=50000, measure_step=1,
        env_name='ALE/Seaquest-v5', horizon=np.inf):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """



    env = gym.make(env_name, render_mode='rgb_array')
    torch.manual_seed(seed)
    env.seed(seed)

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=autoencoder.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    memory = Memory(max_memory_size)

    for episode in range(num_episodes):
        # display the performance
        if episode % measure_step == 0:
            evaluate_step(autoencoder, env)
            wandb.log({"Episode": episode})
            wandb.log({"lr": scheduler.get_lr()[0]})

        state = env.reset()
        # memory.state.append(state)
        img = env.render()
        memory.update(img)

        done = False
        i = 0
        while not done:
        # for i in range(640):
            i += 1
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            # print("done???", done, file=sys.stderr, flush=True)

            if i > horizon:
                done = True

            if i % render_step == 0:
                img = env.render()
                # save state, action, reward sequence
                memory.update(img)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train_step(batch_size, autoencoder, optimizer, scheduler, memory)

        scheduler.step()
        wandb.log({'lr': scheduler.get_last_lr()[0]})

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

monitor = 'Validation MSE'
autoencoder = SlotAttentionAE(**dict_args, resolution=resize)
autoencoder.to(device)

if args.pretrained:
    state_dict = torch.load(args.sa_state_dict)['state_dict']
    # state_dict = torch.load(args.sa_state_dict)
    autoencoder.load_state_dict(state_dict=state_dict, strict=False)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parametersПодозритель
accelerator = args.device
# devices = [int(args.devices)]
gpus = [0]

print(torch.cuda.device_count(), flush=True)

#  precision=16,
# deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None
# else:
#     ckpt = torch.load(args.from_checkpoint)
#
#     autoencoder.load_state_dict(state_dict=ckpt, strict=False)

train_loop(env_name='Seaquest-v0')

wandb.finish()



