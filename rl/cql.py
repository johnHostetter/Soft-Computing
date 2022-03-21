#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:46:02 2021

@author: john
"""

import torch
import numpy as np

def one_hot_encoding(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def build_tensors(list_of_tensors, list_of_indices):
    states, actions, next_states, rewards = [], [], [], []
    for idx in list_of_indices:
        states.append(list_of_tensors[idx][0])
        actions.append(list_of_tensors[idx][1])
        rewards.append(list_of_tensors[idx][2])
        next_states.append(list_of_tensors[idx][3])
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    return states, actions, next_states, rewards

class TabularNetwork(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super(TabularNetwork, self).__init__()
        self.num_states = num_states
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.num_states, num_actions)
        )

    def forward(self, states):
        onehot = one_hot_encoding(states, self.num_states)
        return self.network(onehot)
    
class CQLModel(object):
    def __init__(self, cql_params):
        # discount for future reward
        self.gamma = cql_params['gamma']
        # the alpha parameter used in CQL, value of weight on the CQL coefficient
        self.cql_alpha = cql_params['alpha']
        self.batch_size = cql_params['batch_size']  # the batch size of CQL
        # the number of gradient steps used for projection
        self.number_of_batches = cql_params['batches']
        # the learning rate of CQL
        self.learning_rate = cql_params['learning_rate']
        # the number of CQL iterations to run
        self.number_of_iterations = cql_params['iterations']
        # the action set length of the environment
        self.action_set_length = cql_params['action_set_length']
        
    def build_q_table(self, number_of_states):
        # prepare the Q-table
        self.network = TabularNetwork(number_of_states,
                                      self.action_set_length)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate)
        
    def q_backup_sparse_sampled(self, q_values, state_index, action_index,
                                next_state_index, reward, rule_weights):
        next_state_q_values = q_values[next_state_index, :]
        values = np.max(next_state_q_values, axis=-1)
        target_value = (reward + self.gamma * values)
        return target_value

    def project_qvalues_cql_sampled(self, state_index, action_index, target_values, rule_weights=None):
        # train with a sampled dataset
        target_qvalues = torch.tensor(target_values, dtype=torch.float32)
        state_index = torch.tensor(state_index, dtype=torch.int64)
        action_index = torch.tensor(action_index, dtype=torch.int64)
        pred_qvalues = self.network(state_index)
        logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)

        pred_qvalues = pred_qvalues.gather(
            1, action_index.reshape(-1, 1)).squeeze()
        cql_loss = logsumexp_qvalues - pred_qvalues

        loss = torch.mean((pred_qvalues - target_qvalues)**2)
        # loss = torch.mean(torch.tensor(rule_weights) * ((pred_qvalues - target_qvalues)**2))
        loss = loss + self.cql_alpha * torch.mean(cql_loss)

        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()

        number_of_states = self.get_number_of_rules()
        pred_qvalues = self.network(torch.arange(number_of_states))
        return pred_qvalues.detach().numpy()

    def conservative_q_iteration(self, sampled=False, training_dataset=None, rule_weights=None, **kwargs):
        """
        Runs Conservative Q-iteration.

        Args:
          sampled (bool): Whether to use sampled datasets for training or not.
          training_dataset (list): list of (s, a, r, ns) pairs
          rule_weights (list): list of the rule activations where the i'th observation's current state (s) 
              correponds to the i'th element of rule_weights
        """

        number_of_states = self.get_number_of_rules()
        number_of_actions = self.action_set_length

        q_values = np.zeros((number_of_states, number_of_actions))
        for i in range(self.number_of_iterations):
            if sampled:
                for j in range(self.number_of_batches):
                    training_idx = np.random.choice(np.arange(
                        len(training_dataset)), size=self.batch_size)  # batch size was 256, then 1028
                    state_index, action_index, next_state, reward = build_tensors(
                        training_dataset, training_idx)

                    rule_weights_sample = np.array(rule_weights)[training_idx]

                    target_values = self.q_backup_sparse_sampled(q_values, state_index, action_index,
                                                                 next_state, reward, rule_weights_sample, **kwargs)

                    intermediate_values = self.project_qvalues_cql_sampled(state_index, action_index,
                                                                           target_values, rule_weights=rule_weights_sample)
                    if j == self.number_of_batches - 1:
                        q_values = intermediate_values
            else:
                raise Exception(
                    "The online version of Conservative Fuzzy Q-Learning is not yet available.")
        self.q_table = q_values
        return self.q_table