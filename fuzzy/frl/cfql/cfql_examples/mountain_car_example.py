#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:57:25 2021

@author: john
"""

import os
import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy.frl.fql.fql import FQLModel
from fuzzy.frl.cfql.cfql import CFQLModel
from fuzzy.frl.fql.fis import InputStateVariable, Trapeziums, Gaussian, Build

# GLOBAL_SEED = 1
# LOCAL_SEED = 42
# np.random.seed(GLOBAL_SEED)


class Agent:
    def __init__(self, model):
        self.model = model

    def get_initial_action(self, state):
        try:
            return self.model.get_initial_action(state)
        except AttributeError:
            return self.model.get_action(state)

    def get_action(self, state):
        return self.model.get_action(state)

    def learn(self, state, reward, trajectories):
        return self.model.run(state, reward)


class NewAgent:
    def __init__(self, model):
        self.model = model

    def get_initial_action(self, state):
        try:
            return self.model.get_initial_action(state)
        except AttributeError:
            return self.model.get_action(state)

    def get_action(self, state):
        return self.model.get_action(state)

    def learn(self, state, reward, trajectories):
        return self.model.learn(state, reward, trajectories)


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


def get_fis_from_cfql(cfql):
    antecedents = [[], []]
    for antecedent in cfql.antecedents[0]:
        antecedents[0].append(Gaussian(antecedent['center'], antecedent['sigma']))
    for antecedent in cfql.antecedents[1]:
        antecedents[1].append(Gaussian(antecedent['center'], antecedent['sigma']))
    p = InputStateVariable()
    p.fuzzy_set_list = antecedents[0]
    v = InputStateVariable()
    v.fuzzy_set_list = antecedents[1]
    return Build(p, v)


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
                      ' epsilon=', model.ee_rate, ' best mean eps=', epsilon, ' avg. reward=', np.mean(rewards[-100:]))
            except AttributeError:
                print('EPS=', iteration, ' reward=', r,
                      ' epsilon=', 0.0, ' best mean eps=', epsilon, ' avg. reward=', np.mean(rewards[-100:]))
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

    if render:
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
                  ' epsilon=', 1.0, ' best mean eps=', 1.0, ' avg. reward=', np.mean(rewards[-100:]))
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


def train_env(model=None, max_eps=500, render=False):
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
                  ' epsilon=', model.ee_rate, ' best mean eps=', epsilon, ' avg. reward=', np.mean(rewards[-100:]))
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

    if render:
        plt.figure(figsize=(14, 5))
        plt.plot(best_mean_rewards[1:])
        plt.ylabel('Rewards')
        plt.show()

    return model, trajectories, rewards


from rl.testbeds.mountain_car import MountainCar

output_df = None
MAX_NUM_EPISODES = 10
for LOCAL_SEED in range(10):
    print('----- SEED: {} -----'.format(LOCAL_SEED))
    # --- random play (baseline) ---
    print('Random baseline')
    _, _, random_rewards = random_play_mountain_car(max_eps=MAX_NUM_EPISODES + 1)
    random_df = pd.DataFrame({'policy': ['Random'] * len(random_rewards), 'episode': range(len(random_rewards)), 'total_reward': random_rewards})
    random_df['unsupervised_size'] = 0
    random_df['rules'] = 0

    for FCQL_num_of_train_episodes in [16, 32, 64, 128]:
        np.random.seed(LOCAL_SEED)
        os.environ['PYTHONHASHSEED'] = str(LOCAL_SEED)
        random.seed(LOCAL_SEED)
        np.random.seed(LOCAL_SEED)

        # --- random play training data for CFQL ---

        print('Train data for CFQL')
        _, trajectories, rewards = random_play_mountain_car(max_eps=FCQL_num_of_train_episodes + 1)
        fcql_train_df = pd.DataFrame({'policy': ['FCQL_random_train_data'] * len(rewards), 'episode': range(len(rewards)), 'total_reward': rewards})
        fcql_train_df['unsupervised_size'] = FCQL_num_of_train_episodes
        fcql_train_df['rules'] = 0

        # --- online & static Fuzzy Q-Learning ---

        env, fis = get_fis_env()
        print('Observation shape:', env.observation_space.shape)
        print('Action length:', env.action_space.n)
        action_set_length = env.action_space.n

        # model = FQLModel(gamma=0.99,
        #                  alpha=0.1,
        #                  ee_rate=1.,
        #                  action_set_length=action_set_length,
        #                  fis=fis)
        #
        # mountain_car = MountainCar(LOCAL_SEED, 100 + 1, Agent(model), verbose=True)
        # _, static_fql_rewards = mountain_car.play(True)
        # static_fql_df = pd.DataFrame({'policy': ['online_static_FQL'] * len(static_fql_rewards), 'total_reward': static_fql_rewards})
        # static_fql_df['rules'] = fis.get_number_of_rules()

        # --- online & dynamic CFQL ---

        clip_params = {'alpha': 0.1, 'beta': 0.7}
        fis_params = {'inference_engine': 'product'}
        # note this alpha for CQL is different from CLIP's alpha
        cql_params = {
            'gamma': 0.99, 'alpha': 0.1, 'batch_size': 1028, 'batches': 50,
            'learning_rate': 1e-2, 'iterations': 100, 'action_set_length': action_set_length
        }
        # new_cql_params = {
        #     'gamma': 0.99, 'alpha': 0.1, 'batch_size': 1, 'batches': 1,
        #     'learning_rate': 1e-2, 'iterations': 1, 'action_set_length': action_set_length
        # }

        # print('online CFQL')
        # new_cfql.fit(train_X, trajectories, ecm=True, Dthr=0.01, verbose=True, offline=False)
        # mountain_car = MountainCar(LOCAL_SEED, 100 + 1, NewAgent(new_cfql), verbose=True)
        # _, dynamic_online_cfql_rewards = mountain_car.play(True)
        # dynamic_cfql_df = pd.DataFrame({'policy': ['online_dynamic_CFQL'] * len(dynamic_online_cfql_rewards), 'total_reward': dynamic_online_cfql_rewards})
        # dynamic_cfql_df['rules'] = get_fis_from_cfql(cfql).get_number_of_rules()

        # --- offline & dynamic CFQL
        print('offline CFQL')
        cfql = CFQLModel(clip_params, fis_params, cql_params)
        # new_cfql = CFQLModel(clip_params, fis_params, new_cql_params)
        X = [trajectories[0][0]]
        for idx, trajectory in enumerate(trajectories):
            X.append(trajectory[3])

        train_X = np.array(X)
        cfql.fit(train_X, trajectories, ecm=True, Dthr=0.01, verbose=True)
        mountain_car = MountainCar(LOCAL_SEED, 100 + 1, Agent(cfql), verbose=True)
        # mountain_car.play(False)
        print('Greedy FCQL')
        _, _, _, greedy_offline_rewards = play_mountain_car(cfql, MAX_NUM_EPISODES + 1, False)
        dynamic_fcql_df = pd.DataFrame({'policy': ['offline_dynamic_FCQL'] * len(greedy_offline_rewards), 'episode': range(len(greedy_offline_rewards)), 'total_reward': greedy_offline_rewards})
        dynamic_fcql_df['unsupervised_size'] = FCQL_num_of_train_episodes
        dynamic_fcql_df['rules'] = len(cfql.rules)

        # sometimes, adding a bit of exploration to the exploitation helps CFQL when offline training data is too few
        # print('FCQL + EEP')
        # cfql.ee_rate = 0.15
        # _, _, _, ee_offline_rewards = play_mountain_car(cfql, 100 + 1, False)

        # Online Fuzzy Q-Learning, but this time, with the same partitioning used as the CFQL model
        dynamic_fql = FQLModel(gamma=0.99,
                         alpha=0.1,
                         ee_rate=1.,
                         action_set_length=action_set_length,
                         fis=get_fis_from_cfql(cfql))

        # training FQL online
        print('Training FQL online')
        mountain_car = MountainCar(LOCAL_SEED, MAX_NUM_EPISODES + 1, Agent(dynamic_fql), verbose=True)
        _, training_dynamic_fql_rewards = mountain_car.play(True)
        train_dynamic_fql_df = pd.DataFrame({'policy': ['online_training_dynamic_FQL'] * len(training_dynamic_fql_rewards), 'episode': range(len(training_dynamic_fql_rewards)), 'total_reward': training_dynamic_fql_rewards})
        train_dynamic_fql_df['rules'] = get_fis_from_cfql(cfql).get_number_of_rules()
        train_dynamic_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes

        # testing FQL online
        print('Testing FQL online')
        mountain_car = MountainCar(LOCAL_SEED, MAX_NUM_EPISODES + 1, Agent(dynamic_fql), verbose=True)
        mountain_car.agent.model.ee_rate = 0.0
        _, testing_dynamic_fql_rewards = mountain_car.play(False, exploit=True)
        test_dynamic_fql_df = pd.DataFrame({'policy': ['online_testing_dynamic_FQL'] * len(testing_dynamic_fql_rewards), 'episode': range(len(testing_dynamic_fql_rewards)), 'total_reward': testing_dynamic_fql_rewards})
        test_dynamic_fql_df['rules'] = get_fis_from_cfql(cfql).get_number_of_rules()
        test_dynamic_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes

        seed_df = pd.concat([fcql_train_df, random_df, dynamic_fcql_df, train_dynamic_fql_df, test_dynamic_fql_df])
        seed_df['seed'] = LOCAL_SEED

        if output_df is None:
            output_df = seed_df
        else:
            output_df = pd.concat([output_df, seed_df])

output_df.to_csv('mountain_car_results.csv', index=False)
