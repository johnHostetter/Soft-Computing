#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:56:59 2021

@author: john
"""

# https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f

import os
import sys

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
        rule_activations_during_end = []  # keep track of what rules led to end of episode
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
                    rule_activations_during_end.append(algo.agent.flcs[0].current_rule_activations)
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.std(episode_rewards), rule_activations_during_end

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
        EPOCHS = 12

        seed_df = None
        dataset = dataset[:1000]
        for num_of_train_episodes in range(10, 251, 10):
        # for num_of_train_episodes in [5]:
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
            # train_X = transformer.transform(train_X)

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
            print('CQL Alpha: {}'.format(
                cql_alpha))  # cql alpha 0.5 with batch size 32 and 100 episodes worked well (i.e., 487.95 +- 30.50225401507239)
            offline_mimo = MIMO_replay(transformer, antecedents_, rules_, 2, consequents_, cql_alpha=cql_alpha,
                                       learning_rate=1e-1)

            # # the following specs should solve the cart pole problem
            #
            # # for action 0 (i.e., flcs[0]),
            # flc_0_qs = np.array([4.09273247, 2.99155165, 1.69529502, 0.9554578, 3.38400144, 4.03330933,
            #                      3.94449846, 2.29485375, 1.43395984, 3.49536297, 3.66994179, 2.94600878,
            #                      2.06561499, 3.20644239, 2.88323048, 2.97839352, 2.75879564, 2.44817454,
            #                      3.6129228])
            # # for action 1 (i.e., flcs[1]),
            # flc_1_qs = np.array([4.08707284, 2.95673377, 1.69364686, 0.95616081, 3.32566896, 3.98658785,
            #                      3.99590144, 2.33340544, 1.45550679, 3.55704939, 3.68730338, 3.00891885,
            #                      2.10325604, 3.24888868, 2.88073079, 3.03985276, 2.77599565, 2.34069858,
            #                      3.57920188])

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
                offline_mimo, train_epoch_losses, val_epoch_losses = offline_q_learning(offline_mimo, trajectories,
                                                                                        val_trajectories, EPOCHS,
                                                                                        batch_size,
                                                                                        gamma=0.99)  # gamma was 0.5
                for flc_idx, flc in enumerate(offline_mimo.flcs):
                    print('{}: {}'.format(flc_idx, flc.y))

                for idx, flc in enumerate(offline_mimo.flcs):
                    flc.save('{}'.format(idx))
            else:
                print(flc_0_qs[20], flc_1_qs[20])
                print(rules_[20])
                # flc_0_qs[20] = 17  # 18 also works, 18.5 starts to not work
                offline_mimo.flcs[0].y = flc_0_qs
                offline_mimo.flcs[1].y = flc_1_qs
                for idx, flc in enumerate(offline_mimo.flcs):
                    flc.save('{}'.format(idx))
            ### END OF TRAINING

            from neuro_q_net import EvaluationWrapper

            avg_score, std_score, curr_rules_during_end = evaluate_on_environment(env)(EvaluationWrapper(offline_mimo))
            tmp = np.array([curr_rules[0] for curr_rules in curr_rules_during_end])
            counts = np.unique(tmp.argmax(axis=1), return_counts=True)
            print(counts)
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
