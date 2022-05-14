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
import matplotlib.pyplot as plt

from fuzzy.self_adaptive.make_rules_and_terms import unsupervised

from d3rlpy.algos import DiscreteCQL
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer

# get CartPole dataset
dataset, env = get_cartpole()


def evaluate_on_environment(env, n_trials=100, epsilon=0.0, render=False):
    """ Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env (gym.Env): gym-styled environment.
        n_trials (int): the number of trials.
        epsilon (float): noise factor for epsilon-greedy policy.
        render (bool): flag to render environment.

    Returns:
        callable: scoerer function.


    """

    def scorer(algo, *args):
        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.std(episode_rewards)

    return scorer


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


# seed 10 worked very well (solved), 11, 12, 14 did not work at all (not solved)
# seed 13 worked okay but then got suboptimal (~65.61)
# below was originally here, so, if this doesn't work, add it back in
# SEED = 12
# os.environ['PYTHONHASHSEED'] = str(SEED)
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)


def play_cart_pole(env, model, num_episodes, gamma=0.9,
                   title='DQL', verbose=True):
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
            # except AssertionError:
            except Exception:
                q_values = model.predict(state[np.newaxis, :])
                action = torch.argmax(q_values).item()

                # Take action and add reward to total
                next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            state = next_state
            memory.append((state, action, next_state, reward, done))

        memory.append((state, action, next_state, reward, done))
        final.append(total)
        episodes.append({'trajectory': memory, 'cummulative reward': total})
        # plot_results(final, title)

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
        # plot_results(final,title)
    return memory, final, np.array(states)


def plot_results(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
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


def offline_q_learning(model, training_dataset, validation_dataset, max_epochs=12, batch_size=32, gamma=0.9):
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

SAVE = False
val_loss_df = None
train_loss_df = None
online_evaluation_df = None
for SEED in range(5):
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
    EPOCHS = 3

    seed_df = None
    dataset = dataset[:1000]
    # for num_of_train_episodes in range(100, 1000 + 1, 100):
    for num_of_train_episodes in range(10, 101, 10):
    # for num_of_train_episodes in [100]:
        print('num of training episodes available: {}'.format(num_of_train_episodes))
        # split train and test episodes
        train_episodes, val_episodes = train_test_split(dataset, test_size=0.2)
        train_episodes = train_episodes[:num_of_train_episodes]

        # # start of CQL code
        #
        # # setup CQL algorithm
        # cql = DiscreteCQL(use_gpu=False)
        # cql._alpha = 0.1
        #
        # # start training
        # cql.fit(train_episodes,
        #         eval_episodes=val_episodes,
        #         n_epochs=6,
        #         scorers={
        #             'environment': evaluate_on_environment(env),  # evaluate with CartPol-v0 environment
        #             'advantage': discounted_sum_of_advantage_scorer,  # smaller is better
        #             'td_error': td_error_scorer,  # smaller is better
        #             'value_scale': average_value_estimation_scorer  # smaller is better
        #         })
        #
        # # evaluate
        # print(evaluate_on_environment(env)(cql))
        #
        # # end of CQL code

        trajectories, train_X = transform_data(train_episodes)
        val_trajectories, _ = transform_data(val_episodes)

        from sklearn.preprocessing import Normalizer

        transformer = Normalizer().fit(train_X)

        # get replay results
        from neuro_q_net import MIMO_replay

        rules_, weights_, antecedents_, consequents_ = unsupervised(train_X, None, ecm=True, Dthr=4e-1)
        print('There are {} rules'.format(len(rules_)))

        for input_variable in antecedents_:
            print(len(input_variable))
        # mimo = MIMO_replay(antecedents_, rules_, 2, consequents_, 0., .1)

        # print('online training FQL')
        # episodes, memory, online_train_scores = q_learning(env, mimo, MAX_NUM_EPISODES, gamma=.9, epsilon=1.0,
        #                     replay=True, title='Mamdani Neuro-Fuzzy Q-Network')
        # # online_train_fql_df = pd.DataFrame({'policy': ['online_train_FQL'] * len(online_train_scores),
        # #                                     'episode': range(len(online_train_scores)),
        # #                                     'total_reward': online_train_scores})
        # # online_train_fql_df['rules'] = len(mimo.flcs[0].rules)
        # # online_train_fql_df['unsupervised_size'] = FCQL_num_of_train_episodes
        #
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
        print('CQL Alpha: {}'.format(cql_alpha))  # cql alpha 0.5 with batch size 32 and 100 episodes worked well (i.e., 487.95 +- 30.50225401507239)
        offline_mimo = MIMO_replay(transformer, antecedents_, rules_, 2, consequents_, cql_alpha=0.5,
                                   learning_rate=5e-2)

        batch_size = 128
        offline_mimo, train_epoch_losses, val_epoch_losses = offline_q_learning(offline_mimo, trajectories,
                                                                                val_trajectories, EPOCHS, batch_size,
                                                                                gamma=0.99)  # gamma was 0.5

        from neuro_q_net import EvaluationWrapper

        avg_score, std_score = evaluate_on_environment(env)(EvaluationWrapper(offline_mimo))
        # save the training losses
        loss_df = pd.DataFrame({'policy': ['FCQL'] * len(train_epoch_losses),
                                'epoch': range(len(train_epoch_losses)),
                                'train_loss': train_epoch_losses, 'val_loss': val_epoch_losses})
        loss_df['train_size'] = num_of_train_episodes
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

        if SAVE:
            # save the training losses
            train_loss_fcql_df = pd.DataFrame({'policy': ['offline_FCQL'] * len(train_epoch_losses),
                                               'epoch': range(len(train_epoch_losses)),
                                               'train_loss': train_epoch_losses})
            train_loss_fcql_df['rules'] = len(offline_mimo.flcs[0].rules)
            for idx, input_variable in enumerate(antecedents_):
                train_loss_fcql_df['input_variable_{}'.format(idx)] = len(input_variable)
            train_loss_fcql_df['train_size'] = num_of_train_episodes
            train_loss_fcql_df['seed'] = SEED

            # save the validation losses
            val_loss_fcql_df = pd.DataFrame({'policy': ['offline_FCQL'] * len(val_epoch_losses),
                                             'epoch': range(len(val_epoch_losses)),
                                             'val_loss': val_epoch_losses})
            val_loss_fcql_df['rules'] = len(offline_mimo.flcs[0].rules)
            for idx, input_variable in enumerate(antecedents_):
                val_loss_fcql_df['input_variable_{}'.format(idx)] = len(input_variable)
            val_loss_fcql_df['train_size'] = num_of_train_episodes
            val_loss_fcql_df['seed'] = SEED

            # save the online evaluation scores
            _, _, offline_scores = play_cart_pole(env, offline_mimo, MAX_NUM_EPISODES)
            offline_fcql_df = pd.DataFrame({'policy': ['offline_FCQL'] * len(offline_scores),
                                            'episode': range(len(offline_scores)),
                                            'total_reward': offline_scores})
            offline_fcql_df['rules'] = len(offline_mimo.flcs[0].rules)
            for idx, input_variable in enumerate(antecedents_):
                offline_fcql_df['input_variable_{}'.format(idx)] = len(input_variable)
            offline_fcql_df['train_size'] = num_of_train_episodes
            offline_fcql_df['seed'] = SEED

            # print(np.mean(offline_scores))

            # seed_df = pd.concat([online_train_fql_df, online_test_fql_df, offline_fcql_df])
            # seed_df['seed'] = SEED

            # record all val losses across seeds & num of train episodes avail.
            if val_loss_df is None:
                val_loss_df = val_loss_fcql_df
            else:
                val_loss_df = pd.concat([val_loss_df, val_loss_fcql_df])

            # record all train losses across seeds & num of train episodes avail.
            if train_loss_df is None:
                train_loss_df = train_loss_fcql_df
            else:
                train_loss_df = pd.concat([train_loss_df, train_loss_fcql_df])

            # record all online evaluation scores across seeds & num of train episodes avail.
            if online_evaluation_df is None:
                online_evaluation_df = offline_fcql_df
            else:
                online_evaluation_df = pd.concat([online_evaluation_df, offline_fcql_df])

            # seed_df.to_csv('seed={}_episodes={}.csv'.format(SEED, num_of_train_episodes), index=False)
            val_loss_df.to_csv('./results/val_losses_seed={}.csv'.format(SEED), index=False)
            train_loss_df.to_csv('./results/train_losses_seed={}.csv'.format(SEED), index=False)
            online_evaluation_df.to_csv('./results/online_evaluation_seed={}.csv'.format(SEED), index=False)

    seed_df.to_csv('./results/flc_output_{}.csv'.format(SEED), encoding='utf-8-sig', index=False)

# val_loss_df.to_csv('./results/val_losses.csv', index=False)
# train_loss_df.to_csv('./results/train_losses.csv', index=False)
# online_evaluation_df.to_csv('./results/online_evaluation.csv', index=False)
