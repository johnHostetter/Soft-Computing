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
import pandas as pd

from fuzzy.self_adaptive.make_rules_and_terms import unsupervised

from torch.utils.data import Dataset
from d3rlpy.datasets import get_cartpole
from sklearn.model_selection import train_test_split

### OLD CODE ###
# get CartPole dataset
dataset, env = get_cartpole()

def transform_data(dataset):
    states = []
    transitions = []
    for episode in dataset:
        for transition in episode.transitions:
            done = transition.terminal == 1.0
            states.append(list(transition.observation))
            transitions.append(
                (transition.observation, transition.action, transition.next_observation, transition.reward, done))
    return transitions, np.array(states)


def old_offline_q_learning(model, training_dataset, validation_dataset, max_epochs=12, batch_size=32, gamma=0.9):
    epoch = 0
    threshold = 1e-2
    val_epoch_losses = []
    train_epoch_losses = []
    prev_val_loss = curr_val_loss = 1e10
    while threshold < curr_val_loss <= prev_val_loss and epoch < max_epochs:
        prev_val_loss = curr_val_loss
        train_losses, val_losses = model.replay(training_dataset, batch_size, validation_dataset, gamma, online=False)
        curr_val_loss = val_losses.mean()
        print('epoch {}: avg. train loss = {} & avg. val loss = {}'
              .format(epoch, train_losses.mean(), val_losses.mean()))
        train_epoch_losses.append(train_losses.mean())
        val_epoch_losses.append(val_losses.mean())
        epoch += 1
    return model, train_epoch_losses, val_epoch_losses


class CartPoleDataset(Dataset):
    """ Offline reinforcement learning dataset of Cart Pole. """

    def __init__(self, data, train):
        self.dataset = data
        # self.dataset, env = get_cartpole()
        # if train:
        #     train_episodes, val_episodes = train_test_split(self.dataset, test_size=0.2)
        #     self.dataset = train_episodes[:250]
        #     # self.dataset = self.dataset[:60]
        # else:
        #     self.dataset = self.dataset[1000:]
        self.transitions, self.unique_states = self.transform_data()

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    def transform_data(self):
        states = []
        transitions = []
        for episode in self.dataset:
            for transition in episode.transitions:
                done = transition.terminal == 1.0
                states.append(list(transition.observation))
                value = {'state': transition.observation, 'action': transition.action, 'reward': transition.reward,
                         'next state': transition.next_observation, 'terminal': done}
                transitions.append(value)
        return transitions, np.array(states)


from torch import optim
from fcql import mimoFLC

# class mimoFLC:
#     def __init__(self, n_inputs, n_outputs, antecedents, rules):
#         self.flcs = []
#         self.optimizers = []
#         for flc_idx in range(n_outputs):
#             flc = torchFLC(n_inputs, 1, antecedents, rules)
#             self.flcs.append(flc)
#             self.optimizers.append(optim.Adam(flc.parameters(), lr=1e-4))
#
#     def predict(self, x):
#         output = []
#         for flc in self.flcs:
#             output.append(list(flc.predict(x).detach().numpy()))
#         return torch.tensor(output).T[0]
#
#     def train(self, mode):
#         for flc in self.flcs:
#             flc.train(mode)
#
#     def zero_grad(self):
#         for flc in self.flcs:
#             flc.zero_grad()


from fuzzy.self_adaptive.cart_pole import evaluate_on_environment


def q_learning(env, model, num_episodes, gamma=0.9,
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20,
               title='DQL', double=False,
               n_update=10, soft=False, verbose=True, memory=[]):
    """Deep Q Learning algorithm using the DQN. """

    final = []
    # memory = []
    episodes = []
    sum_total_replay_time = 0

    # for episode_idx in range(num_episodes):
    episode_idx = 0
    continue_loop = episode_idx < num_episodes
    while continue_loop:
        if len(final) >= 100:
            continue_loop = np.mean(final[-100:]) < 195.0 and episode_idx < num_episodes
        else:
            continue_loop = episode_idx < num_episodes

        episode = []

        if double and not soft:
            # Update target network every n_update steps
            if episode_idx % n_update == 0:
                model.target_update()

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
                t0 = time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma=gamma)
                t1 = time.time()
                sum_total_replay_time += (t1 - t0)
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
        # plot_results(final, title)

        if verbose:
            print("episode: {}, total reward: {}, new epsilon: {}, avg. reward of past 100 episodes: {}".format(
                (episode_idx + 1), total, epsilon, np.mean(final[-100:])))
            if replay:
                print("Average replay time:", sum_total_replay_time / (episode_idx + 1))

            if total == 500:
                print('Maximum total reward reached. Terminate further Q-learning.')
                break

        episode_idx += 1

    return episodes, memory, final


if __name__ == "__main__":
    SAVE = False
    policy = 'fcql'
    val_loss_df = None
    train_loss_df = None
    online_evaluation_df = None
    # print('Start at seed {} and end before seed {}'.format(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])))
    # for SEED in range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])):
    for SEED in range(35, 40):
    # for SEED in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
        print('Using seed {}'.format(SEED))
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        env = gym.make('CartPole-v1')
        env.seed(SEED)
        env.action_space.seed(SEED)

        # Number of states
        n_state = env.observation_space.shape[0]
        # Number of actions
        n_action = env.action_space.n
        # Number of episodes
        MAX_NUM_EPISODES = 100  # was 160
        # Learning rate
        lr = 0.001
        # Number of epochs allowed
        EPOCHS = 30

        seed_df = None
        dataset = dataset[:1000]
        for num_of_train_episodes in range(10, 251, 10):
            print('num of training episodes available: {}'.format(num_of_train_episodes))
            # split train and test episodes
            train_episodes, val_episodes = train_test_split(dataset, test_size=0.2)
            train_episodes = train_episodes[:num_of_train_episodes]

            train_data = CartPoleDataset(train_episodes, train=True)
            val_data = CartPoleDataset(val_episodes, train=False)
            trajectories, train_X = transform_data(train_episodes)
            val_trajectories, _ = transform_data(val_episodes)

            # get replay results
            from neuro_q_net import MIMO_replay
            # rules_, weights_, antecedents_, consequents_ = unsupervised(train_data.unique_states, None, ecm=True,
            #                                                             Dthr=4e-1)

            rules_, weights_, antecedents_, consequents_ = unsupervised(train_X, None, ecm=True,
                                                                        Dthr=4e-1)
            print('There are {} rules'.format(len(rules_)))

            for input_variable in antecedents_:
                print(len(input_variable))
            # mimo = MIMO_replay(antecedents_, rules_, 2, consequents_, 0., .1)

            # from fuzzy.common.flc import FLC as torchFLC

            n_outputs = 2
            flc = mimoFLC(len(antecedents_), n_outputs, antecedents_, rules_)

            # print('online training FQL')
            # episodes, memory, online_train_scores = q_learning(env, flc, MAX_NUM_EPISODES, gamma=.9, epsilon=1.0,
            #                     replay=True, title='Mamdani Neuro-Fuzzy Q-Network')
            # online_train_fql_df = pd.DataFrame({'policy': ['online_train_FQL'] * len(online_train_scores),
            #                                     'episode': range(len(online_train_scores)),
            #                                     'total_reward': online_train_scores})
            # online_train_fql_df['rules'] = len(mimo.flcs[0].rules)
            # online_train_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes

            # print('online testing FQL')
            # _, _, online_test_scores = play_cart_pole(env, mimo, 100)
            # print(np.mean(online_test_scores))
            # online_test_fql_df = pd.DataFrame({'policy': ['online_test_FQL'] * len(online_test_scores),
            #                                     'episode': range(len(online_test_scores)),
            #                                     'total_reward': online_test_scores})
            # online_test_fql_df['rules'] = len(mimo.flcs[0].rules)
            # online_test_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes

            from neuro_q_net import DoubleMIMO

            print('offline FCQL')
            # t = 0.1
            # # percent_of_data = num_of_train_episodes / len(dataset)
            # val = (num_of_train_episodes / 10) * np.log(2 + np.sqrt(3)*t)
            # cql_alpha = 1 / (1 + np.exp(val))
            cql_alpha = 0.5
            print('CQL Alpha: {}'.format(
                cql_alpha))  # cql alpha 0.5 with batch size 32 and 100 episodes worked well (i.e., 487.95 +- 30.50225401507239)
            offline_mimo = MIMO_replay(None, antecedents_, rules_, 2, consequents_, cql_alpha=cql_alpha,
                                       learning_rate=1e-4)

            # the following specs will come close to solving (i.e., 381 avg +- 101)
            # for action 0 (i.e., flcs[0]),
            flc_0_qs = np.array([32.23857237, 20.56550323, 27.55988969, 25.99648251, 16.35570309, 18.95074817,
                                 12.4922282, 21.3039934, 11.58842305, 5.90257348, 3.19148588, 30.84851585,
                                 29.50740081, 24.88781145, 29.93867306, 22.45899488, 29.64593866, 27.79719068,
                                 20.13681364, 13.21699227, 15.90490055, 3.14113929, 26.80655194, 25.76440097,
                                 12.9910143, 11.86838104, 7.07246614, 5.06043945, -0.58641758, 10.90387111,
                                 27.27257721, 18.67683394])
            # for action 1 (i.e., flcs[1]),
            flc_1_qs = np.array([32.24291762, 20.01849851, 27.33601231, 26.12146665, 15.51193979, 19.22837849,
                                 12.4818727, 21.71117879, 11.66582381, 5.87689104, 3.17553727, 30.94495598,
                                 29.32600965, 25.56327505, 29.4906996, 22.89623226, 30.27304413, 27.94220619,
                                 20.33806456, 13.44641513, 15.71906733, 3.48654253, 27.14557043, 26.12270416,
                                 13.41508282, 12.44581132, 7.35404242, 5.33562759, -0.53207424, 11.05446885,
                                 27.47501511, 19.20216549])

            ### START OF TRAINING ###

            if True:
                batch_size = 64
                from fuzzy.self_adaptive.fcql import offline_q_learning
                # flc = offline_mimo
                # flc.consequences = torch.nn.Parameter(torch.tensor(np.array([flc_0_qs, flc_1_qs]).T))
                # flc.consequences.requires_grad = False
                flc, train_epoch_losses, val_epoch_losses = offline_q_learning(flc, train_data,
                                                                                        val_data, EPOCHS,
                                                                                        batch_size,
                                                                                        gamma=0.99)  # gamma was 0.5
                # flc, train_epoch_losses, val_epoch_losses = old_offline_q_learning(flc, trajectories,
                #                                                                         val_trajectories, EPOCHS,
                #                                                                         batch_size,
                #                                                                         gamma=0.99)  # gamma was 0.5
                # print(flc.consequences)
            else:
                print(flc_0_qs[20], flc_1_qs[20])
                print(rules_[20])
                # flc_0_qs[20] = 17  # 18 also works, 18.5 starts to not work
                # offline_mimo.flcs[0].y = flc_0_qs
                # offline_mimo.flcs[1].y = flc_1_qs
                flc.consequences = torch.nn.Parameter(torch.tensor(np.array([flc_0_qs, flc_1_qs]).T))
                # offline_mimo.model.consequences = torch.nn.Parameter(torch.tensor(np.array([flc_0_qs, flc_1_qs]).T))
            ### END OF TRAINING

            from neuro_q_net import EvaluationWrapper

            avg_score, std_score, curr_rules_during_end = evaluate_on_environment(env)(flc)
            print((avg_score, std_score))
            # save the training losses

            loss_df = pd.DataFrame({'policy': [policy.upper()] * len(train_epoch_losses),
                                    'epoch': range(len(train_epoch_losses)),
                                    'train_loss': train_epoch_losses, 'val_loss': val_epoch_losses})
            loss_df['train_size'] = num_of_train_episodes
            loss_df['transitions'] = len(trajectories)
            loss_df['rules'] = len(offline_mimo.flcs[0].rules)
            for idx, input_variable in enumerate(antecedents_):
                loss_df['input_variable_{}'.format(idx)] = len(input_variable)
            loss_df['avg_score'] = avg_score
            loss_df['std_score'] = std_score
            loss_df['seed'] = SEED
            print(loss_df.head())

            if seed_df is None:
                seed_df = loss_df
            else:
                seed_df = pd.concat([seed_df, loss_df])

            # the following specs should solve the cart pole problem

            # for action 0 (i.e., flcs[0]),
            flc_0_qs = np.array([0.01797587, 0.01548789, 0.01624808, 0.01517777, 0.01444484,
                                 0.01334789, 0.01096193, 0.01500386, 0.01313595, 0.00786661,
                                 0.01302805, 0.00839448, 0.00506901, 0.01796555, 0.01735124,
                                 0.01529711, 0.01492031, 0.01733455])
            # for action 1 (i.e., flcs[1]),
            flc_1_qs = np.array([0.01794827, 0.01504194, 0.01596201, 0.01499904, 0.0139775,
                                 0.01357084, 0.01123002, 0.01522818, 0.01342868, 0.00802537,
                                 0.01320628, 0.00850744, 0.00512792, 0.01793559, 0.01715578,
                                 0.01545542, 0.01472266, 0.01750127])

        # seed_df.to_csv('./results/flc_output_{}.csv'.format(SEED), encoding='utf-8-sig', index=False)
