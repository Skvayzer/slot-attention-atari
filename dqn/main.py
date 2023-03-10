import copy
import os
import sys
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from torchvision.utils import save_image

from ale_py.roms import Seaquest

from ale_py import ALEInterface
ale = ALEInterface()
ale.loadROM(Seaquest)

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    print('Training started', file=sys.stderr, flush=True)

    for episode in range(n_episodes):
        print('Episode', episode, file=sys.stderr, flush=True)

        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            # print('t:', t, file=sys.stderr, flush=True)

            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)
            # print(obs, reward, done, info,
            #       file=sys.stderr, flush=True)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        # if episode % 20 == 0:
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward), file=sys.stderr, flush=True)
        wandb.log({'Episode': episode,
                   'Total reward': total_reward,
                   'Steps done': steps_done})
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    # env = gym.wrappers.Monitor(env, '/home/sa_atari/videos/' + 'dqn_sequest_video_test')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward), file=sys.stderr, flush=True)
                break

    env.close()
    return

def reinitialize_memory(env, n_episodes, max_memory_size=20000):
    print('Memory reinitialization started', file=sys.stderr, flush=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            # print('t:', t, file=sys.stderr, flush=True)

            action = policy_net(state.to('cuda')).max(1)[1].view(1,1)

            obs, reward, done, info = env.step(action)
            # print(obs, reward, done, info,
            #       file=sys.stderr, flush=True)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            if len(memory) >= max_memory_size:
                return
            state = next_state


            if done:
                break
        # if episode % 20 == 0:
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Memory Size: {}'.format(steps_done, episode, t, total_reward, len(memory)), file=sys.stderr, flush=True)

    # env.close()
    return


def generate_memory(env, episodes=20, max_memory_size=20000, mode='train'):
    # memory = Memory(max_memory_size)
    # env = FrameStack(env, 4)
    i = 0
    for e in range(episodes):
        print(f"Memory Episode {e}", file=sys.stderr, flush=True)

        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0

        # img = env.render()
        done = False
        count = 0
        lives = env.unwrapped.ale.lives()
        while not done:
            count += 1
            # memory.update(state)
            action = policy_net(state.to('cuda')).max(1)[1].view(1,1)

            obs, reward, done, _ = env.step(action)
            state = get_state(obs)
            total_reward += reward



            # if i % 4 == 0:
            #     continue
            # print("state shape", state.shape, file=sys.stderr, flush=True)
            if count > 3:
                img = torch.tensor(env.render(mode="rgb_array")).permute(2, 0, 1) / 255
                save_image(img, os.path.join("/mnt/data/users_data/smirnov/sa_atari/datasets/seaquest_purified", mode, mode + '_' + str(i) + '.png'))
                # img = env.render()
                i += 1
                print('Total images: {} \t Episode: {} \t Total reward: {} \t'.format(i, e,
                                                                                    total_reward),
                  file=sys.stderr, flush=True)

            current_lives = env.unwrapped.ale.lives()
            if current_lives < lives:
                count = 0

            if i > max_memory_size:
                return

    return #memory

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb.init(project='atari test')
    
    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY



    steps_done = 0

    # create environment
    env = gym.make("Seaquest-v4")
    env = make_env(env)

    # create networks
    policy_net = DQN(n_actions=env.action_space.n).to(device)
    ckpt = torch.load("/home/sa_atari/dqn_seaquest_model_40000 (new)")
    policy_net.load_state_dict(ckpt['state_dict'])
    target_net = DQN(n_actions=env.action_space.n).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # optimizer.load_state_dict(ckpt['optimizer'])
    # # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    # reinitialize_memory(env, 100000, max_memory_size=MEMORY_SIZE)
    
    # # train model
    # train(env, 120000)
    #
    # state = {
    #     'epoch': 120000,
    #     'state_dict': policy_net.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # torch.save(state, "/home/sa_atari/dqn_seaquest_model_120000")
    # policy_net = torch.load("/home/sa_atari/dqn_seaquest_model_40000")




    train_memory = generate_memory(env, episodes=70000, max_memory_size=50, mode='train')
    # train_memory.preprocess()
    # np.savez("/home/sa_atari/seaquest_train", images=train_memory.images)

    val_memory = generate_memory(env, episodes=15000, max_memory_size=50, mode='val')
    # test(env, 2000, policy_net, render=False)

