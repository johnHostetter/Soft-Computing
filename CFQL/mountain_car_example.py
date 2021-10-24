#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:57:25 2021

@author: john
"""

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from fql import FQLModel
from cfql import CFQLModel
from cfql import conservative_q_iteration
from fis import InputStateVariable, Trapeziums, Build

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

def play_mountain_car(model, max_eps=100):
    env, _ = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

    q_diff = model.q_table.max(axis=1) - model.q_table.min(axis=1)

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
        
        q_values = model.q_table[np.argmax(model.R)]
        if np.max(q_values) < np.median(q_diff):
            action_index = random.randint(0, model.action_set_length - 1)
        
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
    
    return model, np.array(visited_states), trajectories, rewards

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


# if __name__ == '__main__':
#     model = train_env(max_eps=500)

model, trajectories, _ = train_env(max_eps=60)

env, fis = get_fis_env()
print('Observation shape:', env.observation_space.shape)
print('Action length:', env.action_space.n)
action_set_length = env.action_space.n
cfql = CFQLModel(gamma=0.99, alpha=0.1, ee_rate=1., action_set_length=action_set_length, fis=fis)

rule_weights = []
transformed_trajectories = []
for trajectory in trajectories:
    state = trajectory[0]
    action_index = trajectory[1]
    reward = trajectory[2]
    next_state = trajectory[3]
    done = trajectory[4]
    
    # this code will only use the rule with the highest degree of activation to do updates
    # cfql.truth_value(state)
    # rule_index = np.argmax(cfql.R)
    # rule_index_weight = cfql.R[rule_index]
    # rule_weights.append(rule_index_weight)
    # cfql.truth_value(next_state)
    # next_rule_index = np.argmax(cfql.R)
    # next_rule_index_weight = cfql.R[next_rule_index]
    # transformed_trajectories.append((rule_index, action_index, reward, next_rule_index, done))
    
    # this code will use all rules that are activated greater than epsilon 
    epsilon = 0.1
    cfql.truth_value(state)
    next_rule_index = np.argmax(cfql.R)
    next_rule_index_weight = cfql.R[next_rule_index]
    indices_of_interest = np.where(np.array(cfql.R) > epsilon)[0]
    for rule_index in indices_of_interest:
        truth_value = cfql.R[rule_index]
        rule_weights.append(truth_value)
        transformed_trajectories.append((rule_index, action_index, reward, next_rule_index, done))

print('num. of trajectories: %d' % len(trajectories))
print('num. of transformed trajectories: %d' % len(transformed_trajectories))

env.num_states = cfql.fis.get_number_of_rules()
env.num_actions = env.action_space.n
q_values = conservative_q_iteration(env, cfql.network,
                                    num_itrs=100, project_steps=100, discount=0.95, cql_alpha=0.9, 
                                    weights=None, render=True,
                                    sampled=True,
                                    training_dataset=transformed_trajectories, rule_weights=rule_weights)
env, fis = get_fis_env()
print('Observation shape:', env.observation_space.shape)
print('Action length:', env.action_space.n)
action_set_length = env.action_space.n
offline_cfql = FQLModel(gamma=0.0, alpha=0.0, ee_rate=0., action_set_length=action_set_length, fis=fis)
offline_cfql.q_table = q_values
# q_diff = q_values.max(axis=1) - q_values.min(axis=1)
# median_q_diff = np.median(q_diff)

# exploit the learned policy
_, _, _, offline_rewards = play_mountain_car(offline_cfql)

# exploit the learned policy, but add a small bit of randomness (performs better than greedy)
offline_cfql = FQLModel(gamma=0.0, alpha=0.0, ee_rate=.2, action_set_length=action_set_length, fis=fis)
offline_cfql.q_table = q_values
_, _, offline_rewards = train_env(offline_cfql, 100)