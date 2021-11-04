#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:58:04 2021

@author: john
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from fuzzy import InputStateVariable
from fuzzy import Trapeziums, Gaussian
from fuzzy import Build
from nfqn import NeuroFuzzyQNetwork as FQLModel

GLOBAL_SEED = 1
LOCAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Define membership functions for MountainCar problems
def get_fis_env():
    # p = InputStateVariable(Trapeziums(-1.2, -1.2, -1.2, -0.775),
    #                        Trapeziums(-1.2, -0.775, -0.775, -0.35),
    #                        Trapeziums(-0.775, -0.35, -0.35, 0.075),
    #                        Trapeziums(-0.35, 0.075, 0.075, 0.5),
    #                        Trapeziums(0.075, 0.5, 0.5, 0.5))
    # v = InputStateVariable(Trapeziums(-0.07, -0.07, -0.07, -0.035),
    #                        Trapeziums(-0.07, -0.035, -0.035, 0.),
    #                        Trapeziums(-0.035, 0., 0., 0.035),
    #                        Trapeziums(0., 0.035, 0.035, 0.07),
    #                        Trapeziums(0.035, 0.035, 0.035, 0.07))
    p = InputStateVariable(Gaussian(-1.2, 0.425),
                           Gaussian(-0.775, 0.425),
                           Gaussian(-0.35, 0.274),
                           Gaussian(0.075, 0.425),
                           Gaussian(0.5, 0.425))
    v = InputStateVariable(Gaussian(-0.07, 0.035),
                           Gaussian(-0.035, 0.035),
                           Gaussian(0.0, 0.035),
                           Gaussian(0.035, 0.035),
                           Gaussian(0.035, 0.035))
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    fis = Build(p, v)
    env.seed(LOCAL_SEED)
    return env, fis

def play_mountain_car(model, max_eps=100):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

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
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            iteration += 1
            r = 0
                
        # render the environment for the last couple episodes
        if False and iteration + 1 > (max_eps - 5):
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
    print('Epsilon=', model.ee_rate)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()
    
    env.close()
    
    return model, np.array(visited_states), trajectories

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
    visited_states = []
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value = env.reset()
            visited_states.append(state_value)
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
                
        # render the environment for the last couple episodes
        if iteration + 1 > (max_eps - 3):
            env.render()
        
        prev_state = state_value
        state_value, reward, done, _ = env.step(action)
        visited_states.append(state_value)
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
    
    env.close()
    
    return model, np.array(visited_states), trajectories

# if __name__ == '__main__':
#     model = train_env(max_eps=150)

model, visited_states, trajectories = train_env(max_eps=150) # originally was 500 episodes

# testing that CLIP works for defining the input state variables' terms

from clip import CLIP

X = visited_states[-15000:]
antecedents = CLIP(X, X, X.min(axis=0), X.max(axis=0), terms=[], alpha=0.1, beta=0.7, theta=1e-8)

input_variables = []
for i in range(len(antecedents)):
    terms = []
    for j in range(len(antecedents[i])):
        terms.append(Gaussian(antecedents[i][j]['center'], antecedents[i][j]['sigma']))
    variable = InputStateVariable()
    variable.fuzzy_set_list = terms
    input_variables.append(variable)
    
clip_fis = Build()
clip_fis.list_of_input_variable = input_variables
clip_fql = FQLModel(gamma=0.99, alpha=0.1, ee_rate=1., action_set_length=3, fis=clip_fis)
clip_model, _, _ = train_env(clip_fql, max_eps=50)

# testing that offline works for Fuzzy Rule-Based Q-Learning

offline_fis = Build()
offline_fis.list_of_input_variable = input_variables # use the input variables obtained using CLIP
offline_fql = FQLModel(gamma=1.0, alpha=1e-4, ee_rate=0., action_set_length=3, fis=offline_fis)
offline_fql.cql_alpha = 0.4
new_episode = True
import random
for epoch_idx in range(1):
    print(epoch_idx)
    random.shuffle(trajectories)
    for trajectory in trajectories[:70000]:
        state = trajectory[0]
        action_index = trajectory[1]
        reward = trajectory[2]
        next_state = trajectory[3]
        done = trajectory[4]
        if new_episode:
            offline_fql.get_initial_offline_action(state, action_index)
        offline_fql.offline_run(state, action_index, reward)
        new_episode = done
    
played_model, _, _ = play_mountain_car(offline_fql, max_eps=50)