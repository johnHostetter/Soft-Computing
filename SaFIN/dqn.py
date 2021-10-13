#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 22:30:33 2021

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
    
class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))
        
# Expand DQL class with a replay function.
class DQN_replay(DQN):
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
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            # update q values
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            self.update(states.tolist(), all_q_values.tolist())
            


class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self, memory): # ATTN: memory input not required, done for convenience
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets)


class fuzzy_DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = SaFIN(alpha=0.1, beta=0.9)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        return self.target.predict([s])
        
    def target_update(self, memory):
        ''' Update target network with the model weights.'''
        if len(memory) > 0:
            X = np.array([memory[i][0] for i in range(len(memory))])
            Y = self.predict(X).numpy()
            size = int(np.round(X.shape[0] / 2))
            # self.target.fit(X, Y, batch_size=min(size, 200), epochs=1, verbose=True, shuffle=False)
            self.target.fit(X, Y, batch_size=int(np.round(X.shape[0] / 2)), epochs=1, verbose=True, shuffle=False)
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory[-5000:], size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    if len(self.target.rules) == 0:
                        q_values_next = self.predict(next_state)
                        q_values[action] = reward + gamma * torch.max(q_values_next).item()
                    else:
                        q_values_next = self.target_predict([next_state])
                    # q_values_next = self.target_predict([next_state])
                        q_values[action] = reward + gamma * np.max(q_values_next)

                targets.append(q_values)

            self.update(states, targets)
            
# Expand DQL class with a replay function.
class DQN_replay_w_distill(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.student = SaFIN(0.1, 0.9) # 0.1, 0.9 get decent results sometimes
        
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
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            # update q values
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            self.update(states.tolist(), all_q_values.tolist())
            
            if len(memory) % 100 == 0:
                self.policy_distillation(memory)
            
    def policy_distillation(self, memory):
        # train the student model after replay with memory
        limited_memory = memory[-100:]
        # limited_memory = memory
        X = np.array([limited_memory[i][0] for i in range(len(limited_memory))])
        Y = self.predict(X).numpy()
        # size = int(np.round(X.shape[0] / 2))
        size = 100
        # self.target.fit(X, Y, batch_size=min(size, 200), epochs=1, verbose=True, shuffle=False)
        self.student.fit(X, Y, batch_size=size, epochs=1, verbose=True, shuffle=True, rule_pruning=True)
        
class DQN_double_distill(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = copy.deepcopy(self.model)
        self.student = SaFIN(0.2, 0.95) # 0.01, 0.99 get decent results sometimes
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self, memory): # ATTN: memory input not required, done for convenience
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets)
            
            # if len(memory) % 5000 == 0:
            # if len(memory) % 10000 == 0:
            if False:
                self.policy_distillation(memory)
            
    def policy_distillation(self, memory):
        # train the student model after replay with memory
        limited_memory = memory[-2000:]
        # limited_memory = memory
        X = np.array([limited_memory[i][0] for i in range(len(limited_memory))])
        Y = self.target_predict(X).numpy()
        # Y = (Y - Y.min()) / (Y.max() - Y.min()) # try normalizing the Q values; this worked 500+, and okay the one time ~150
        # Y = Y - Y.mean() / (Y.max() - Y.min()) # try standardizing the Q values; this worked 500+, and okay the one time ~150

        size = int(np.round(X.shape[0] / 2))
        # size = 100
        # self.target.fit(X, Y, batch_size=min(size, 200), epochs=1, verbose=True, shuffle=False)
        self.student.fit(X, Y, batch_size=500, epochs=1, verbose=True, shuffle=True, rule_pruning=False)