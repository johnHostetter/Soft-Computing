#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:57:25 2021

@author: john
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from fuzzy.frl.fql.fql import FQLModel
from fuzzy.frl.cfql.cfql import CFQLModel
from fuzzy.frl.fql.fis import InputStateVariable, Trapeziums, Build

GLOBAL_SEED = 1
LOCAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Define membership functions for MountainCar problems
def get_fis_env():
    p = InputStateVariable(Trapeziums(-1.2, -1.2, -1.2, -0.775),
                           Trapeziums(-1.2, -0.775, -0.775, -0.35),
                           Trapeziums(-0.775, -0.35, -0.35, 0.075),
                           Trapeziums(-0.35, 0.075, 0.075, 0.5),
                           Trapeziums(0.075, 0.5, 0.5, 0.5))
    v = InputStateVariable(Trapeziums(-0.07, -0.07, -0.07, -0.035),
                           Trapeziums(-0.07, -0.035, -0.035, 0.),
                           Trapeziums(-0.035, 0., 0., 0.035),
                           Trapeziums(0., 0.035, 0.035, 0.07),
                           Trapeziums(0.035, 0.035, 0.035, 0.07))
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    fis = Build(p, v)
    env.seed(LOCAL_SEED)
    return env, fis

def play_mountain_car(model, max_eps=100, render=False):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)

    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    visited_states = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            visited_states.append(state_value)
            action = model.get_action(state_value)
            rewards.append(r)
            mean_reward = np.mean(rewards[-50:])
            best_mean_rewards.append(mean_reward)
            if len(best_mean_rewards) > 2:
                epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
            else:
                epsilon = 0
            try:
                print('EPS=', iteration, ' reward=', r,
                      ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            except AttributeError:
                print('EPS=', iteration, ' reward=', r,
                      ' epsilon=', 0.0, ' best mean eps=', epsilon)
            iteration += 1
            r = 0

        # render the environment for the last couple episodes
        if render and iteration + 1 > (max_eps - 6):
            env.render()

        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        visited_states.append(state_value)
        trajectories.append((prev_state, action, reward, state_value, done))
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = model.get_action(state_value)

        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print(model.q_table)
    try:
        print('Epsilon=', model.ee_rate)
    except AttributeError:
        print('Epsilon=', 0.0)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()

    env.close()

    return model, np.array(visited_states), trajectories, rewards

def random_play_mountain_car(model=None, max_eps=500):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)

    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            action = env.action_space.sample()
            rewards.append(r)
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', 1.0, ' best mean eps=', 1.0)
            iteration += 1
            r = 0

        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        trajectories.append((prev_state, action, reward, state_value, done))
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = env.action_space.sample()
        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print('Epsilon=', 1.0)
    return model, trajectories, rewards

def train_env(model=None, max_eps=500):
    env, fis = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

    # Create Model
    if model is None:
        model = FQLModel(gamma=0.99,
                         alpha=0.1,
                         ee_rate=1.,
                         action_set_length=action_set_length,
                         fis=fis)
    rewards = []
    r = 0
    done = True
    iteration = 0
    trajectories = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            action = model.get_initial_action(state_value)
            rewards.append(r)
            mean_reward = np.mean(rewards[-50:])
            best_mean_rewards.append(mean_reward)
            if len(best_mean_rewards) > 2:
                epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
            else:
                epsilon = 0
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            iteration += 1
            r = 0
            # Epsilon decay
            model.ee_rate -= model.ee_rate * 0.01
            if model.ee_rate <= 0.2:
                model.ee_rate = 0.2
        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        trajectories.append((prev_state, action, reward, state_value, done))
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = model.run(state_value, reward)
        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print(model.q_table)
    print('Epsilon=', model.ee_rate)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()
    return model, trajectories, rewards

# 10 episodes also works, but some interactions will still require ~2000 time-steps
model, trajectories, _ = train_env(max_eps=10)
# _, trajectories, _ = random_play_mountain_car(max_eps=100)

env, fis = get_fis_env()
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
new_cfql = CFQLModel(clip_params, fis_params, cql_params)
X = [trajectories[0][0]]
for idx, trajectory in enumerate(trajectories):
    X.append(trajectory[3])

train_X = np.array(X)
cfql.fit(train_X, trajectories, ecm=True, Dthr=0.01, verbose=True)
_, _, _, greedy_offline_rewards = play_mountain_car(cfql, 100, False)
cfql.ee_rate = 0.15
_, _, _, ee_offline_rewards = play_mountain_car(cfql, 100, False)
