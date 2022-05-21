#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:27:21 2021

@author: john
"""

import os
import gym
import time
import torch
import random
import numpy as np
from fuzzy.denfis.ecm import ECM
from fuzzy.self_adaptive.clip import CLIP, rule_creation

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

env = gym.make('LunarLander-v2')  # requires 'pip install Box2D'


def unsupervised(train_X, trajectories, ecm=False, Dthr=1e-3, verbose=False):
    """
    Trains the CFQLModel with its AdaptiveNeuroFuzzy object on the provided training data, 'train_X',
    and their corresponding trajectories.

    Parameters
    ----------
    train_X : 2-D Numpy array
        The input vector, has a shape of (number of observations, number of inputs/attributes).
    trajectories : list
        A list containing elements that have the form of (state, action, reward, next state, done).
        The 'state' and 'next state' items are 1D Numpy arrays that have the shape of (number of inputs/attributes,).
        The 'action' item is an integer that references the index of the action chosen when in 'state'.
        The 'reward' item is a float that describes the immediate reward received after taking 'action' in 'state'.
        The 'done' item is a boolean that is True if this list element is the end of an episode, False otherwise.
    ecm : boolean, optional
        This boolean controls whether to enable the ECM algorithm for candidate rule generation. The default is False.
    Dthr : float, optional
        The distance threshold for the ECM algorithm; only matters if ECM is enabled. The default is 1e-3.
    verbose : boolean, optional
        If enabled (True), the execution of this function will print out step-by-step to show progress. The default is False.

    Returns
    -------
    None.

    """
    print('The shape of the training data is: (%d, %d)\n' %
          (train_X.shape[0], train_X.shape[1]))
    train_X_mins = train_X.min(axis=0)
    train_X_maxes = train_X.max(axis=0)

    # this Y array only exists to make the rule generation simpler
    dummy_Y = np.zeros(train_X.shape[0])[:, np.newaxis]
    Y_mins = np.array([-1.0])
    Y_maxes = np.array([1.0])

    if verbose:
        print('Creating/updating the membership functions...')

    alpha = 0.1
    beta = 0.7

    start = time.time()
    antecedents = CLIP(train_X, dummy_Y, train_X_mins, train_X_maxes,
                       [], alpha=alpha, beta=beta)
    end = time.time()
    if verbose:
        print('membership functions for the antecedents generated in %.2f seconds.' % (
                end - start))

    start = time.time()
    consequents = CLIP(dummy_Y, train_X, Y_mins, Y_maxes, [],
                       alpha=alpha, beta=beta)
    end = time.time()
    if verbose:
        print('membership functions for the consequents generated in %.2f seconds.' % (
                end - start))

    if ecm:
        if verbose:
            print('\nReducing the data observations to clusters using ECM...')
        start = time.time()
        clusters = ECM(train_X, [], Dthr)
        if verbose:
            print('%d clusters were found with ECM from %d observations...' % (
                len(clusters), train_X.shape[0]))
        reduced_X = [cluster.center for cluster in clusters]
        reduced_dummy_Y = dummy_Y[:len(reduced_X)]
        end = time.time()
        if verbose:
            print('done; the ECM algorithm completed in %.2f seconds.' %
                  (end - start))
    else:
        reduced_X = train_X
        reduced_dummy_Y = dummy_Y

    if verbose:
        print('\nCreating/updating the fuzzy logic rules...')
    start = time.time()
    antecedents, consequents, rules, weights = rule_creation(reduced_X, reduced_dummy_Y,
                                                             antecedents,
                                                             consequents,
                                                             [],
                                                             [],
                                                             problem_type='SL',
                                                             consistency_check=False)

    K = len(rules)
    end = time.time()
    if verbose:
        print('%d fuzzy logic rules created/updated in %.2f seconds.' %
              (K, end - start))
    return rules, weights, antecedents, consequents


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
                q_values = model.predict([state])
                action = np.argmax(q_values.numpy())
            # take action and extract results
            next_state, reward, done, _ = env.step(action)  # take a random action
            # update reward
            total += reward
            trajectories.append((state, action, next_state, reward, done))
            states.append(state.tolist())
            if done:
                trajectories.append((state, action, next_state, reward, done))
                states.append(next_state.tolist())
                break
            # add to the final reward
            final.append(total)
    env.close()
    return trajectories, final, np.array(states)


action_set_length = env.action_space.n
trajectories, random_rewards, train_X = lunar_lander(env)
val_trajectories, _, val_X = lunar_lander(env)


# X = [trajectories[0][0]]
# for idx, trajectory in enumerate(trajectories):
#     X.append(trajectory[3])
#
# train_X = np.array(X)
clip_params = {'alpha': 0.1, 'beta': 0.7}
fis_params = {'inference_engine': 'product'}
# note this alpha for CQL is different than CLIP's alpha
cql_params = {
    'gamma': 0.99, 'alpha': 0.1, 'batch_size': 1028, 'batches': 50,
    'learning_rate': 1e-2, 'iterations': 100, 'action_set_length': action_set_length
}
# cfql = CFQLModel(clip_params, fis_params, cql_params)
# cfql.fit(train_X, trajectories, ecm=True, Dthr=0.125, prune_rules=False, apfrb_sensitivity_analysis=False, )
# cfql_trajectories, cfql_rewards, _ = lunar_lander(env, cfql)

# get replay results
try:
    from neuro_q_net import MIMO_replay
except ModuleNotFoundError:
    from fuzzy.self_adaptive.neuro_q_net import MIMO_replay

rules_, weights_, antecedents_, consequents_ = unsupervised(train_X, None)
print('There are {} rules'.format(len(rules_)))
# mimo = MIMO_replay(antecedents_, rules_, 2, consequents_, 0., .1)


# replay = q_learning(env, mimo, episodes, gamma=.9, epsilon=1.0,
#                     replay=True, title='Mamdani Neuro-Fuzzy Q-Network')

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


from neuro_q_net import DoubleMIMO

num_of_actions = 4
from sklearn.preprocessing import Normalizer

transformer = Normalizer().fit(train_X)
offline_mimo = MIMO_replay(transformer, antecedents_, rules_, num_of_actions, consequents_, cql_alpha=0.5,
                           learning_rate=1e-4)
EPOCHS = 12
batch_size = 64
offline_mimo, train_epoch_losses, val_epoch_losses = offline_q_learning(offline_mimo, trajectories,
                                                                        val_trajectories, EPOCHS, batch_size,
                                                                        gamma=0.99)  # gamma was 0.5
_, offline_scores, _ = lunar_lander(env, offline_mimo)
print(offline_scores)
print('avg.: {}'.format(np.mean(offline_scores)))
