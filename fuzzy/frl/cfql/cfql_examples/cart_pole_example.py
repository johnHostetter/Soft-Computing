#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:23:36 2021

@author: john
"""

import os
import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from fuzzy.frl.cfql.cfql import CFQLModel

SEED = 10
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def play_cart_pole(env, model, num_episodes, gamma=0.0,
               title = 'CQL', verbose=True):
    global FUZZY
    """Deep Q Learning algorithm using the DQN. """

    final = []
    episode_i = 0
    episodes = []

    for episode in range(num_episodes):
        episode_i += 1
        memory = []

        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            try:
                action = model.get_action(state[np.newaxis, :])
                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)
            except AssertionError:
                action = model.get_action(state)

                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            state = next_state
            memory.append((state, action, reward, next_state, done))

        memory.append((state, action, reward, next_state, done))
        final.append(total)
        episodes.append({'trajectory':memory, 'cummulative reward':total})
        plot_results(final, title)

        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))

    return episodes, memory, final

def plot_results(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

def random_search_cart_pole(env, num_episodes, title='Random Strategy'):
    """ Random search strategy implementation."""
    final = []
    states = []
    memory = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Sample random actions
            action = env.action_space.sample()
            # Take action and extract results
            next_state, reward, done, _ = env.step(action)
            # Update reward
            total += reward
            memory.append((state, action, reward, next_state, done))
            states.append(state)
            if done:
                memory.append((state, action, reward, next_state, done))
                states.append(next_state)
                break
        # Add to the final reward
        final.append(total)
        plot_results(final, title)
    return memory, final, np.array(states)

env = gym.make('CartPole-v1')
env.seed(SEED)
env.action_space.seed(SEED)

# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
n_episodes = 15
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001

# Get replay results
trajectories, _, states = random_search_cart_pole(env, n_episodes)

print('Observation shape:', env.observation_space.shape)
print('Action length:', env.action_space.n)
action_set_length = env.action_space.n
clip_params = {'alpha':0.1, 'beta':0.7}
fis_params = {'inference_engine':'product'}
# note this alpha for CQL is different than CLIP's alpha
cql_params = {
    'gamma':0.99, 'alpha':0.1, 'batch_size':1028, 'batches':50,
    'learning_rate':1e-2, 'iterations':100 ,'action_set_length':action_set_length
    }

cfql = CFQLModel(clip_params, fis_params, cql_params)
X = [trajectories[0][0]]
for idx, trajectory in enumerate(trajectories):
    X.append(trajectory[3])

train_X = np.array(X)
cfql.fit(train_X, trajectories, ecm=True, Dthr=0.01, verbose=True)
_, _, greedy_offline_rewards = play_cart_pole(env, cfql, 100)
cfql.ee_rate = 0.15
_, _, ee_offline_rewards = play_cart_pole(env, cfql, 100)
