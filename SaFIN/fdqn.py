#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:50:48 2021

@author: john
"""

import gym
import copy
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
# from PIL import Image
# from IPython.display import clear_output
# import math
# import torchvision.transforms as T
import numpy as np

try:
    from SaFIN.safin import SaFIN
except ImportError:
    from safin import SaFIN

class FuzzyDQN():
    ''' A Self Organizing Mamdani Neuro-Fuzzy Q-Network '''
    def __init__(self, env, X_mins, X_maxes):
        # X_mins = list(env.observation_space.low)
        # X_mins[1] = -10
        # X_mins[3] = -10
        # X_maxes = list(env.observation_space.high)
        # X_maxes[1] = 10
        # X_maxes[3] = 10
        # self.model = SaFIN(0.0005, 0.7, X_mins, X_maxes)
        self.model = SaFIN(0.6, 0.7, 'RL', X_mins, X_maxes)
    
    def update(self, state, y): 
        if state.ndim == 1 and y.ndim == 1:
            rmse = self.model.fit(np.array([state]), np.array([y]), batch_size=1, epochs=1, l_rate=0.01, rule_pruning=False, 
                           shuffle=False, verbose=False)
        elif state.ndim == 2 and y.ndim == 2:
            rmse = self.model.fit(state, y, batch_size=20, epochs=1, l_rate=0.05, rule_pruning=False, 
                           shuffle=True, verbose=False)
        
    def predict(self, state):
        if self.model.K == 0:
            ACTION_DIM = 2
            Y = np.array([[0.0] * ACTION_DIM])
            rmse = self.model.fit(np.array([state]), Y, batch_size=1, epochs=1, l_rate=1.0, rule_pruning=False, gradient_descent=False,
                           shuffle=False, verbose=False)
        q_values = self.model.predict(state)
        return torch.Tensor(q_values[0])
    
# Expand FuzzyDQN class with a replay function.
class FuzzyDQN_replay(FuzzyDQN):
    def replay(self, memory, size, gamma=0.9):
        """New replay function"""
        if len(memory)>=size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) # transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = np.array(states)
            actions_tensor = torch.Tensor(actions)
            next_states = np.array(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = torch.Tensor(self.model.predict(states)) # predicted q_values of all states
            all_q_values_next = torch.Tensor(self.model.predict(next_states))
            # update q values
            all_q_values[range(len(all_q_values)),actions] = rewards + gamma * torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()] = rewards[is_dones_indices.tolist()]
        
            self.update(states, all_q_values.numpy())