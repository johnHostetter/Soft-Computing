#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:56:26 2021

@author: john

    This code demonstrates the Conservative Fuzzy Rule-Based Q-Learning Algorithm.
    
    It is corrected from the FQL code at the following link: 
        https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning
    
    I have extended it with Tabular Conservative Q-Learning with code from:
        https://sites.google.com/view/offlinerltutorial-neurips2020/home
    
"""

import copy
import torch
import random
import operator
import itertools
import functools
import numpy as np

from fis import Build

GLOBAL_SEED = 1
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

def one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

class TabularNetwork(torch.nn.Module):
    def __init__(self, num_states, num_actions):
      super(TabularNetwork, self).__init__()
      self.num_states = num_states
      self.network = torch.nn.Sequential(
          torch.nn.Linear(self.num_states, num_actions)
      )

    def forward(self, states):
        onehot = one_hot(states, self.num_states)
        return self.network(onehot)

def get_tensors(list_of_tensors, list_of_indices):
    s, a, ns, r = [], [], [], []
    for idx in list_of_indices:
        s.append(list_of_tensors[idx][0])
        a.append(list_of_tensors[idx][1])
        r.append(list_of_tensors[idx][2])
        ns.append(list_of_tensors[idx][3])
    s = np.array(s)
    a = np.array(a)
    ns = np.array(ns)
    r = np.array(r)
    return s, a, ns, r

class CFQLModel(object):
    def __init__(self, gamma, l_rate, ee_rate,
                 action_set_length, fis=Build()):
        self.current_rule_activations = []
        self.past_rule_activations = []
        self.each_rules_chosen_action = []
        self.state_values = []
        self.state_Qs = []
        self.error = 0
        self.gamma = gamma # discount factor
        self.l_rate = l_rate # learning rate
        self.ee_rate = ee_rate
        self.exploration_exploitation_rate = ee_rate # higher values result in more exploration
        self.action_set_length = action_set_length
        self.fis = fis

        self.network = TabularNetwork(self.fis.get_number_of_rules(),
                                 self.action_set_length)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-1)

    # Fuzzify to get the degree of truth values
    def truth_value(self, state_value):
        self.current_rule_activations = []
        L = []
        input_variables = self.fis.list_of_input_variable
        for index, variable in enumerate(input_variables):
            m_values = []
            fuzzy_sets = variable.get_fuzzy_sets()
            for fuzzy_set in fuzzy_sets:
                membership_value = fuzzy_set.membership_value(state_value[index])
                m_values.append(membership_value)
            L.append(m_values)

        # Calculate Truth Values
        # results are the product of membership functions
        for element in itertools.product(*L):
            self.current_rule_activations.append(functools.reduce(operator.mul, element, 1))

    def action_selection(self):
        self.each_rules_chosen_action = []
        r = random.uniform(0, 1)

        for rule_index in range(self.fis.get_number_of_rules()):
            # Act randomly
            if r < self.exploration_exploitation_rate:
                action_index = random.randint(0, self.action_set_length - 1)
            # Get maximum values
            else:
                state = np.array([rule_index])
                state_tensor = torch.tensor(state, dtype=torch.int64)
                q_values = self.network(state_tensor)
                q_values = q_values.detach().numpy() # detach from PyTorch
                action_index = np.argmax(q_values)
            self.each_rules_chosen_action.append(action_index)

        # 1. Action = sum of truth values*action selection
        # action = 0
        # for index, val in enumerate(self.current_rule_activations):
        #     action += self.each_rules_chosen_action[index]*val
        # action = int(action)
        # if action >= self.action_set_length:
        #         action = self.action_set_length - 1
        action = self.each_rules_chosen_action[np.argmax(self.current_rule_activations)]
        return action
    
    def offline_action_selection(self, q_table, action_index):
        self.each_rules_chosen_action = []
        
        for rule in q_table:
            self.each_rules_chosen_action.append(action_index)
        action = self.each_rules_chosen_action[np.argmax(self.current_rule_activations)]
        return action

    # Q(s,a) = Sum of (degree_of_truth_values[i]*q[i, a])
    def calculate_q_value(self, q_table):
        q_curr = 0
        for rule_index, truth_value in enumerate(self.current_rule_activations):
            # state = np.array([rule_index])
            # state_tensor = torch.tensor(state, dtype=torch.int64)
            # q_values = self.network(state_tensor)
            # q_values = q_values.detach().numpy() # detach from PyTorch
            q_curr += truth_value * q_table[0, self.each_rules_chosen_action[rule_index]]
        self.state_Qs.append(q_curr)
        return q_curr

    # V'(s) = sum of (degree of truth values*max(q[i, a]))
    def calculate_state_value(self, q_table):
        # v_curr = 0
        # for index, rule in enumerate(self.q_table):
        #     v_curr += (self.current_rule_activations[index] * max(rule))
        # self.state_values.append(v_curr)
        v_curr = 0
        for rule_index in range(self.fis.get_number_of_rules()):
            # state = np.array([rule_index])
            # state_tensor = torch.tensor(state, dtype=torch.int64)
            # q_values = self.network(state_tensor)
            # q_values = q_values.detach().numpy() # detach from PyTorch
            q_values = q_table[rule_index]
            v_curr += (self.current_rule_activations[rule_index] * q_values.max())
        self.state_values.append(v_curr)
        return v_curr

    # Q(i, a) += beta*degree_of_truth*delta_Q
    # delta_Q = reward + gamma*V'(s) - Q(s, a)
    def update_q_value(self, reward): # THIS STILL NEEDS UPDATED
        self.error = reward + self.gamma * self.state_values[-1] - self.state_Qs[-1]
        self.q_table = self.network(torch.arange(self.fis.get_number_of_rules())).detach().numpy()
        # self.past_rule_activations is the degree of truth values for the previous state
        for index, truth_value in enumerate(self.past_rule_activations):
            delta_q = self.l_rate * (self.error * truth_value)
            self.q_table[index][self.each_rules_chosen_action[index]] += delta_q
        return self

    def save_state_history(self):
        self.past_rule_activations = copy.copy(self.current_rule_activations)

    def get_initial_action(self, state):
        self.state_values.clear()
        self.state_Qs.clear()
        self.truth_value(state)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_action(self, state):
        self.truth_value(state)
        # numerator = (np.array(self.current_rule_activations)[:,np.newaxis] * self.q_table).sum(axis=0)
        # denominator = np.array(self.current_rule_activations).sum()
        # action = np.argmax(numerator / denominator)
        action = np.argmax((np.array(self.current_rule_activations)[:,np.newaxis] * self.q_table).mean(axis=0))
        # action = self.action_selection()
        return action

    # online execution of FQL
    def run(self, state, reward):
        self.truth_value(state)
        self.calculate_state_value()
        self.update_q_value(reward)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action
    
    #@title Tabular Q-iteration
    
    # --- used for online Q-Learning ---
    
    # def q_backup_sparse(self, env, q_values, discount=0.99):
    #     dS = env.num_states
    #     dA = env.num_actions
          
    #     new_q_values = np.zeros_like(q_values)
    #     value = np.max(q_values, axis=1)
    #     for s in range(dS):
    #         for a in range(dA):
    #             new_q_value = 0
    #             for ns, prob in env.transitions(s, a).items():
    #                 new_q_value += prob * (env.reward(s,a,ns) + discount*value[ns])
    #             new_q_values[s,a] = new_q_value
    #     return new_q_values

    def q_backup_sparse_sampled(self, q_values, states, action_indices, 
                                next_states, rewards, discount=0.99):
        target_values = []
        for idx, state in enumerate(states):
            action_index = action_indices[idx]
            next_state = next_states[idx]
            reward = rewards[idx]
            self.state_values.clear()
            self.state_Qs.clear()
            self.truth_value(next_state) # calculating the degree of applicability for each rule given the current state
            next_rule_activations = self.current_rule_activations
            next_state_value = self.calculate_state_value(q_values) # calculates the value of the next state
            
            self.state_values.clear()
            self.state_Qs.clear()
            self.truth_value(state) # calculating the degree of applicability for each rule given the current state
            current_rule_activations = self.current_rule_activations
            self.offline_action_selection(q_values, action_index)
            this_state_q_value = self.calculate_q_value(q_values)
            
            # using these two values follows the CQL algorithm, whereas using the above values follows the FQL algorithm
            this_state_q_value = 0
            next_state_value = np.max(q_values[np.argmax(next_rule_activations), :], axis=-1)
            
            error = reward + discount * (next_state_value - this_state_q_value)
            target_values.append(error)
            # q_table = self.network(torch.arange(self.fis.get_number_of_rules())).detach().numpy()
            # # self.past_rule_activations is the degree of truth values for the previous state
            # for index, truth_value in enumerate(self.past_rule_activations):
            #     delta_q = self.l_rate * ( * truth_value)
            #     self.q_table[index][self.each_rules_chosen_action[index]] += delta_q
        return target_values
        
            
        # next_state_indices = next_states
        # next_state_q_values = q_values[next_state_indices, :]
        # values = np.max(next_state_q_values, axis=-1)
        # target_value = (rewards + discount * values)
        # return target_value
    
    #@title Conservative Q-Learning
    
    # --- used for online Q-Learning---
    
    # def project_qvalues_cql(self, q_values, network, optimizer, num_steps=50, cql_alpha=0.1, weights=None):
    #     # regress onto q_values (aka projection)
    #     q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
    #     for _ in range(num_steps):
    #         # Eval the network at each state
    #         pred_qvalues = network(torch.arange(q_values.shape[0]))
    #         if weights is None:
    #             loss = torch.mean((pred_qvalues - q_values_tensor)**2)
    #         else:
    #             loss = torch.mean(weights*(pred_qvalues - q_values_tensor)**2)
        
    #     # Add cql_loss
    #     # You can have two variants of this loss, one where data q-values
    #     # also maximized (CQL-v2), and one where only the large Q-values 
    #     # are pushed down (CQL-v1) as covered in the tutorial
    #     cql_loss = torch.logsumexp(pred_qvalues, dim=-1, keepdim=True) # - pred_qvalues
    #     loss = loss + cql_alpha * torch.mean(weights * cql_loss)
    #     network.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     return pred_qvalues.detach().numpy()
    
    def project_qvalues_cql_sampled(self, states, action_indices, target_values, 
                                    cql_alpha=0.1, num_steps=50, weights=None):
        # train with a sampled dataset
        
        # self.q_table = self.network(torch.arange(self.fis.get_number_of_rules())).detach().numpy()
        # # self.past_rule_activations is the degree of truth values for the previous state
        # for index, truth_value in enumerate(self.past_rule_activations):
        #     delta_q = self.l_rate * (self.error * truth_value)
        #     self.q_table[index][self.each_rules_chosen_action[index]] += delta_q
        
        target_values_array = []
        state_indices = []
        action_indices_array = []
        for idx, state in enumerate(states): # where state is the continuous input
            action_index = action_indices[idx]
            error = target_values[idx]
            self.truth_value(state) # calculating the degree of applicability for each rule given the current state
            current_rule_activations = np.array(self.current_rule_activations)
            # current_rule_activations_indices = np.where(current_rule_activations > 0.0)[0]
            current_rule_activations_indices = [np.argmax(current_rule_activations)]
            for rule_index in current_rule_activations_indices:
                truth_value = current_rule_activations[rule_index]
                state_indices.append(rule_index)
                action_indices_array.append(action_index)
                target_values_array.append(truth_value * error)
        
        target_qvalues = torch.tensor(target_values_array, dtype=torch.float32)
        state_indices = torch.tensor(state_indices, dtype=torch.int64)
        action_indices = torch.tensor(action_indices_array, dtype=torch.int64)
        pred_qvalues = self.network(state_indices)
        logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)
        
        pred_qvalues = pred_qvalues.gather(1, action_indices.reshape(-1,1)).squeeze()
        cql_loss = logsumexp_qvalues - pred_qvalues
        
        loss = torch.mean((pred_qvalues - target_qvalues)**2)
        loss = loss + cql_alpha * torch.mean(cql_loss)
    
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        number_of_states = self.fis.get_number_of_rules()
        pred_qvalues = self.network(torch.arange(number_of_states))
        return pred_qvalues.detach().numpy()
    
    def conservative_q_iteration(self, num_itrs=100, project_steps=50, cql_alpha=0.1,
                                 render=False, weights=None, sampled=False,
                                 training_dataset=None, rule_weights=None, **kwargs):
        """
        Runs Conservative Q-iteration.
        
        Args:
          env: A GridEnv object.
          num_itrs (int): Number of FQI iterations to run.
          project_steps (int): Number of gradient steps used for projection.
          cql_alpha (float): Value of weight on the CQL coefficient.
          render (bool): If True, will plot q-values after each iteration.
          sampled (bool): Whether to use sampled datasets for training or not.
          training_dataset (list): list of (s, a, r, ns) pairs
        """
        
        number_of_states = self.fis.get_number_of_rules()
        number_of_actions = self.action_set_length
        
        weights_tensor = None
        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        q_values = np.zeros((number_of_states, number_of_actions))
        for i in range(num_itrs):
            print(i)
            if sampled:
                for j in range(project_steps):
                    training_idx = np.random.choice(np.arange(len(training_dataset)), size=256) # was 256
                    states, action_indices, next_states, rewards = get_tensors(training_dataset, training_idx)
                    
                    # rule_weights_sample = np.array(rule_weights)[training_idx]
                    
                    target_values = self.q_backup_sparse_sampled(q_values, states, action_indices, 
                                                                 next_states, rewards, **kwargs)
                    
                    intermediate_values = self.project_qvalues_cql_sampled(states, action_indices, 
                                                                           target_values, cql_alpha=cql_alpha, weights=None)
                    if j == project_steps - 1:
                        q_values = intermediate_values
            else:
                raise Exception("The online version of Conservative Fuzzy Q-Learning is not yet available.")
                # target_values = q_backup_sparse(env, q_values, **kwargs)
                # q_values = project_qvalues_cql(target_values, network, optimizer,
                #                           weights=weights_tensor,
                #                           cql_alpha=cql_alpha,
                #                           num_steps=project_steps)
          # if render:
          #   plot_sa_values(env, q_values, update=True, title='Q-values Iteration %d' %i)
        self.q_table = q_values
        return self.q_table