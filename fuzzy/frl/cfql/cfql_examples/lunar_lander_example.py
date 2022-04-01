#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:27:21 2021

@author: john
"""

import os
import gym
import torch
import random
import numpy as np

from fuzzy.frl.cfql.cfql import CFQLModel

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

env = gym.make('LunarLander-v2')  # requires 'pip install Box2D'


def lunar_lander(env, model=None):
    env = gym.make('LunarLander-v2')

    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

    final = []
    states = []
    trajectories = []
    n_episodes = 100
    for episode_idx in range(n_episodes):
        print(episode_idx)
        state = env.reset()
        done = False
        total = 0
        while not done:
            if model is not None and episode_idx > (n_episodes - 10):
                env.render()
            if model is None:
                # sample random actions
                action = env.action_space.sample()
            else:
                q_values = model.infer(state)
                action = np.argmax(q_values)
            # take action and extract results
            next_state, reward, done, _ = env.step(action)  # take a random action
            # update reward
            total += reward
            trajectories.append((state, action, reward, next_state, done))
            states.append(state.tolist())
            if done:
                trajectories.append((state, action, reward, next_state, done))
                states.append(next_state.tolist())
                break
            # add to the final reward
            final.append(total)
    env.close()
    return trajectories, final, np.array(states)


action_set_length = env.action_space.n
trajectories, random_rewards, _ = lunar_lander(env)

X = [trajectories[0][0]]
for idx, trajectory in enumerate(trajectories):
    X.append(trajectory[3])

train_X = np.array(X)
clip_params = {'alpha': 0.1, 'beta': 0.7}
fis_params = {'inference_engine': 'product'}
# note this alpha for CQL is different than CLIP's alpha
cql_params = {
    'gamma': 0.99, 'alpha': 0.1, 'batch_size': 1028, 'batches': 50,
    'learning_rate': 1e-2, 'iterations': 100, 'action_set_length': action_set_length
}
cfql = CFQLModel(clip_params, fis_params, cql_params)
cfql.fit(train_X, trajectories, ecm=True, Dthr=0.125, prune_rules=False, apfrb_sensitivity_analysis=False, )
cfql_trajectories, cfql_rewards, _ = lunar_lander(env, cfql)
