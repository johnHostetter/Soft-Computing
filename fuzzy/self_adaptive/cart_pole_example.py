#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:56:59 2021

@author: john
"""

# https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f

import os
import gym
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from fuzzy.self_adaptive.safin import SaFIN

# seed 10 worked very well (solved), 11, 12, 14 did not work at all (not solved)
# seed 13 worked okay but then got suboptimal (~65.61)
SEED = 10
os.environ['PYTHONHASHSEED']=str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# from torch.autograd import Variable
# import random
# from PIL import Image
# from IPython.display import clear_output
# import math
# # import torchvision.transforms as T
# import numpy as np

def play_cart_pole(env, model, num_episodes, gamma=0.9,
               title = 'DQL', verbose=True):
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
                q_values = model.predict(state[np.newaxis, :])
                action = np.argmax(q_values)
                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)
            except AssertionError:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            state = next_state
            memory.append((state, action, next_state, reward, done))

        memory.append((state, action, next_state, reward, done))
        final.append(total)
        episodes.append({'trajectory':memory, 'cummulative reward':total})
        plot_results(final, title)

        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))

    return episodes, memory, final

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
            memory.append((state, action, next_state, reward, done))
            states.append(state)
            if done:
                memory.append((state, action, next_state, reward, done))
                states.append(next_state)
                break
        # Add to the final reward
        final.append(total)
        plot_results(final,title)
    return memory, final, np.array(states)

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

def q_learning(env, model, num_episodes, gamma=0.9,
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20,
               title = 'DQL', double=False,
               n_update=10, soft=False, verbose=True, memory=[]):
    global FUZZY
    """Deep Q Learning algorithm using the DQN. """

    final = []
    # memory = []
    episodes = []
    sum_total_replay_time=0

    for episode_idx in range(num_episodes):
        episode = []

        if double and not soft:
            # Update target network every n_update steps
            if episode_idx % n_update == 0:
                if FUZZY:
                    model.target_update(memory)
                else:
                    model.target_update(None)

        if double and soft:
            model.target_update()

        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            episode.append(memory[-1])
            q_values = model.predict(state).tolist()

            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                # elif replay:
                #     # my own addition
                #     if model.model.K > 200:
                #         model.simplify(memory, replay_size, gamma)
                break

            if replay:
                t0=time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
            else:
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(state, q_values)

            state = next_state

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.05)
        final.append(total)
        # memory.append({'trajectory':memory, 'cummulative reward':total})
        episodes.append(episode)
        plot_results(final, title)

        if verbose:
            print("episode: {}, total reward: {}".format((episode_idx + 1), total))
            if replay:
                print("Average replay time:", sum_total_replay_time / (episode_idx + 1))

            if total == 500:
                print('Maximum total reward reached. Terminate further Q-learning.')
                break

    return episodes, memory, final

env = gym.make('CartPole-v1')
env.seed(SEED)
env.action_space.seed(SEED)

# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes = 160
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001

from fdqn import FuzzyDQN, FuzzyDQN_replay

# fuzzy_dqn = FuzzyDQN()
# episodes, memory, _ =  q_learning(env, fuzzy_dqn, 160, gamma=.95,
#                               epsilon=0.5, replay=False, double=False,
#                               title='A Self Organizing, Adaptive, Mamdani Neuro-Fuzzy Q-Network for Discrete Action',
#                               n_update=5)

# Get replay results
mem, _, states = random_search_cart_pole(env, 100)
X_mins = states.min(axis=0)
X_maxes = states.max(axis=0)
fdqn_replay = FuzzyDQN_replay(env, X_mins, X_maxes)
replay = q_learning(env, fdqn_replay, episodes, gamma=.5, epsilon=0.99, replay=True, replay_size=20,
                    title='Training A Self-Adaptive Mamdani Neuro-Fuzzy Q-Network for Discrete Action + Replay')
_, _, _ = play_cart_pole(env, fdqn_replay, 30, title='Self-Adaptive Mamdani Neuro-Fuzzy Q-Network for Discrete Action + Replay')

quit

# Get DQN results
from dqn import DQN, DQN_replay, DQN_double, fuzzy_DQN_double, DQN_replay_w_distill, DQN_double_distill

# simple_dqn = DQN(n_state, n_action, n_hidden, lr)
# simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3)

# # Get replay results
# dqn_replay = DQN_replay(n_state, n_action, n_hidden, lr)
# replay = q_learning(env, dqn_replay,
#                     episodes, gamma=.9,
#                     epsilon=0.2, replay=True,
#                     title='DQL with Replay')

# Get replay results
# FUZZY = False
# dqn_double = DQN_double(n_state, n_action, n_hidden, lr)
# double =  q_learning(env, dqn_double, episodes, gamma=.95,
#                     epsilon=0.5, replay=True, double=True,
#                     title='Double DQL with Replay', n_update=10)

# Get fuzzy double DQN replay results
# FUZZY = True
# fuzzy_dqn_double = fuzzy_DQN_double(n_state, n_action, n_hidden, lr)
# fuzzy_double =  q_learning(env, fuzzy_dqn_double, episodes, gamma=.95,
#                     epsilon=0.5, replay=True, double=True,
#                     title='Fuzzy Double DQL with Replay', n_update=5)

# DQN + replay + offline policy distillation
for i in range(1):
    FUZZY = True
    ddqn_replay = DQN_double(n_state, n_action, n_hidden, lr)
    episodes, memory, _ =  q_learning(env, ddqn_replay, episodes, gamma=.95,
                                  epsilon=0.5, replay=True, double=True,
                                  title='DDQL with Replay + policy distillation', n_update=5)

    _, _, _ = play_cart_pole(env, ddqn_replay, 10, title='Deep Q-Network (Teacher) + Replay')

    safin = SaFIN(0.3, 0.7)

    # train the student model after replay with memory
    limited_memory = memory[-4000:]
    # limited_memory = memory
    X = np.array([limited_memory[i][0] for i in range(len(limited_memory))])
    Y = ddqn_replay.predict(X).numpy()
    # Y = (Y - Y.min()) / (Y.max() - Y.min()) # try normalizing the Q values; this worked 500+, and okay the one time ~150
    # Y = Y - Y.mean() / (Y.max() - Y.min()) # try standardizing the Q values; this worked 500+, and okay the one time ~150

    size = int(np.round(X.shape[0] / 2))
    size = 500
    safin.fit(X, Y, batch_size=size, epochs=10, verbose=True, shuffle=True, rule_pruning=False)

    _, _, _ = play_cart_pole(env, safin, 30, title='SaFIN (Student) Q-Network')
