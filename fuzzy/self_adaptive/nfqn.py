#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 00:58:02 2021

@author: john
"""

import time
import torch
import random
import operator
import itertools
import functools
import numpy as np

from copy import copy, deepcopy

from fuzzy.self_adaptive.fuzz import Build
from fuzzy.self_adaptive.clip import CLIP, rule_creation
from fuzzy.neuro.adaptive import AdaptiveNeuroFuzzy

class NeuroFuzzyQNetwork(AdaptiveNeuroFuzzy):
    """
        A Self-Adaptive Mamdani Neuro-Fuzzy Q-Network.

        Layer 1 consists of the input (variable) nodes.
        Layer 2 is the antecedent nodes.
        Layer 3 is the rule nodes.
        Layer 4 consists of the consequent nodes.
        Layer 5 is the output (variable) nodes.

        The input vector is denoted as:
            x = (x_1, ..., x_p, ..., x_P)

        The corresponding desired output vector is denoted as:
            d = (d_1, ..., d_q, ..., d_Q),

        while the computed output is denoted as:
            y = (y_1, ..., y_q, ..., y_Q)

        The notations used are the following:

        $P$: number of input dimensions
        $Q$: number of output dimensions
        $I_{p}$: $p$th input node
        $O_{q}$: $q$th output node
        $J_{p}$: number of fuzzy clusters in $I_{p}$
        $L_{q}$: number of fuzzy clusters in $O_{q}$
        $A_{j_p}$: $j$th antecedent fuzzy cluster in $I_{p}$
        $C_{l_q}$: $l$th consequent fuzzy cluster in $O_{q}$
        $K$: number of fuzzy rules
        $R_{k}$: $k$th fuzzy rule

        WARNING: The order in which these functions are called matters.
        Use extreme care when using these functions, as some rely on the neuro-fuzzy network
        having already processed some kind of input (e.g. they may make a reference to self.f3 or similar).
    """
    def __init__(self, gamma, alpha, ee_rate,
                 action_set_length, fis=Build()):
        self.R = []
        self.R_ = []
        self.M = []
        self.V = []
        self.Q = []
        self.Error = 0
        self.gamma = gamma
        self.alpha = alpha
        self.ee_rate = ee_rate
        self.action_set_length = action_set_length
        self.fis = fis

        self.q_table = np.zeros((self.fis.get_number_of_rules(),
                                 self.action_set_length))
    # def __init__(self, alpha=0.2, beta=0.6, X_mins=None, X_maxes=None):
    #     """


    #     Parameters
    #     ----------
    #     alpha : TYPE, optional
    #         DESCRIPTION. The default is 0.2.
    #     beta : TYPE, optional
    #         DESCRIPTION. The default is 0.6.

    #     Returns
    #     -------
    #     None.

    #     """
    #     super().__init__()
    #     self.alpha = alpha # the alpha threshold for the CLIP algorithm
    #     self.beta = beta # the beta threshold for the CLIP algorithm
    #     self.problem_type = 'RL'

    #     self.X_mins = X_mins
    #     self.X_maxes = X_maxes
    #     self.Y_mins = np.array([0., 0.])
    #     self.Y_maxes = np.array([1., 1.])

    def __deepcopy__(self, memo):
        rules = deepcopy(self.rules)
        weights = deepcopy(self.weights)
        antecedents = deepcopy(self.antecedents)
        consequents = deepcopy(self.consequents)
        nfqn = NeuroFuzzyQNetwork(deepcopy(self.alpha), deepcopy(self.beta))
        nfqn.import_existing(rules, weights, antecedents, consequents)
        nfqn.X_mins = deepcopy(self.X_mins)
        nfqn.X_maxes = deepcopy(self.X_maxes)
        if self.P is not None:
            nfqn.orphaned_term_removal()
            nfqn.preprocessing()
            nfqn.update()
        return nfqn

    def truth_value(self, state_value):
        self.R = []
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
            self.R.append(functools.reduce(operator.mul, element, 1))
        # self.R = self.f3 # get the truth values of the fuzzy rules, which are stored in self.f3
        return self # done for easy integration with existing Fuzzy Rule-Based Q Learning

    def action_selection(self):
        self.M = []
        r = random.uniform(0, 1)

        for rule in self.q_table:
            # act randomly
            if r < self.ee_rate:
                action_index = random.randint(0, self.action_set_length - 1)
            # get maximum values
            else:
                action_index = np.argmax(rule)
            self.M.append(action_index)
        action = self.M[np.argmax(self.R)]
        return action

    def offline_action_selection(self, action_index):
        self.M = []

        for rule in self.q_table:
            self.M.append(action_index)
        action = self.M[np.argmax(self.R)]
        return action

    # Q(s, a) = sum of (degree_of_truth_values[i]*q[i,a])
    def calculate_q_value(self):
        q_curr = 0
        for index, truth_value in enumerate(self.R):
            q_curr += truth_value * self.q_table[index, self.M[index]]
        self.Q.append(q_curr)

    # V'(s) = sum of (degree of truth values * max(q[i, a])
    def calculate_state_value(self):
        v_curr = 0
        for index, rule in enumerate(self.q_table):
            v_curr += (self.R[index] * max(rule))
        self.V.append(v_curr)

    # Q(i, a) += beta*degree_of_truth*delta_Q
    # delta_Q = reward + gamma*V'(s) - Q(s, a)
    def update_q_value(self, reward):
        self.Error = reward + self.gamma * self.V[-1] - self.Q[-1]
        # self.R_ is the degree of truth values for the previous state
        for index, truth_value in enumerate(self.R_):
            delta_q = self.alpha * (self.Error * truth_value)
            self.q_table[index][self.M[index]] += delta_q
        return self

    def offline_update_q_value(self, reward):
        self.Error = reward + self.gamma * self.V[-1] - self.Q[-1]
        # self.R_ is the degree of truth values for the previous state
        for index, truth_value in enumerate(self.R_):
            delta_q = self.alpha * (self.Error * truth_value)
            cql_loss = torch.logsumexp(torch.Tensor(self.q_table[index]), dim=-1, keepdim=True)
            delta_q = delta_q + self.cql_alpha * torch.mean(cql_loss).detach().numpy()
            self.q_table[index][self.M[index]] += delta_q
        return self

    def save_state_history(self):
        self.R_ = copy(self.R)

    def get_initial_action(self, state):
        self.V.clear()
        self.Q.clear()
        self.truth_value(state)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_action(self, state):
        self.truth_value(state)
        action = self.action_selection()
        return action

    def run(self, state, reward):
        self.truth_value(state)
        self.calculate_state_value()
        self.update_q_value(reward)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_initial_offline_action(self, state, action_index):
        self.V.clear()
        self.Q.clear()
        self.truth_value(state)
        action = self.offline_action_selection(action_index)
        self.calculate_q_value()
        self.save_state_history()
        return action

    def offline_run(self, state, action_index, reward):
        self.truth_value(state)
        self.calculate_state_value()
        self.offline_update_q_value(reward)
        action = self.offline_action_selection(action_index)
        self.calculate_q_value()
        self.save_state_history()
        return action

    def predict(self, X):
        # update the stored minimums and maximums as new data comes in
        if self.X_mins is None and self.X_maxes is None:
            self.X_mins = X[0]
            self.X_maxes = X[0]
            X_mins = self.X_mins
            X_maxes = self.X_maxes
        else:
            X_mins = X.min(axis=0)
            X_maxes = X.max(axis=0)
            self.X_mins = np.minimum(self.X_mins, X_mins)
            self.X_maxes = np.maximum(self.X_maxes, X_maxes)
        q_values = super(self.__class__, self).predict(X)
        self.Y_mins = np.minimum(self.Y_mins, q_values.min(axis=0))
        self.Y_maxes = np.maximum(self.Y_maxes, q_values.max(axis=0))
        return q_values

    def fit(self, X, Y, batch_size=None, epochs=1, l_rate=0.001, verbose=False, shuffle=True):
        if self.P is None:
            self.P = X.shape[1]
        if self.Q is None:
            self.Q = Y.shape[1]

        if batch_size is None:
            batch_size = 1
        NUM_OF_BATCHES = round(X.shape[0] / batch_size)

        for epoch in range(epochs):
            if shuffle:
                training_data = list(zip(X, Y))
                random.shuffle(training_data)
                shuffled_X, shuffled_Y = zip(*training_data)
                shuffled_X, shuffled_Y = np.array(shuffled_X), np.array(shuffled_Y)
            else:
                shuffled_X, shuffled_Y = X, Y

            for i in range(NUM_OF_BATCHES):
                print('--- Epoch %d; Batch %d ---' % (epoch + 1, i + 1))
                batch_X = X[batch_size*i:batch_size*(i+1)]
                batch_Y = Y[batch_size*i:batch_size*(i+1)]
                if self.X_mins is not None and self.X_maxes is not None:
                    X_mins = self.X_mins
                    X_maxes = self.X_maxes
                else:
                    X_mins = np.min(batch_X, axis=0)
                    X_maxes = np.max(batch_X, axis=0)
                    Y_mins = np.min(batch_Y, axis=0)
                    Y_maxes = np.max(batch_Y, axis=0)
                    if self.X_mins is None and self.X_maxes is None:
                        self.X_mins, self.X_maxes, self.Y_mins, self.Y_maxes = X_mins, X_maxes, Y_mins, Y_maxes
                    else:
                        try:
                            self.X_mins = np.min([self.X_mins, X_mins], axis=0)
                            self.X_maxes = np.min([self.X_maxes, X_maxes], axis=0)
                            self.Y_mins = np.min([self.Y_mins, Y_mins], axis=0)
                            self.Y_maxes = np.min([self.Y_maxes, Y_maxes], axis=0)
                        except AttributeError or TypeError:
                            self.X_mins, self.X_maxes, self.Y_mins, self.Y_maxes = X_mins, X_maxes, Y_mins, Y_maxes

                if self.K > 0:
                    # need to call predict before backpropagation
                    self.predict(batch_X)
                    consequent_delta_c, consequent_delta_widths = self.backpropagation(batch_X, batch_Y)

                    # self.term_dict['consequent_centers'] -= l_rate * np.reshape(consequent_delta_c.mean(axis=0), self.term_dict['consequent_centers'].shape)
                    # adjust the array to match the self.term_dict
                    max_array_size = max(self.L.values())
                    tmp = np.empty((self.Q, max_array_size))
                    tmp[:] = np.nan

                    start = 0
                    # avg_consequent_delta_c = consequent_delta_c.mean(axis=0)
                    avg_consequent_delta_c = np.max(consequent_delta_c, axis=0)
                    for q in range(self.Q):
                        end = start + self.L[q]
                        tmp[q, :self.L[q]] = avg_consequent_delta_c[start:end]
                        start = end

                    self.term_dict['consequent_centers'] += l_rate * tmp

                    # adjust the array to match the self.term_dict
                    max_array_size = max(self.L.values())
                    tmp = np.empty((self.Q, max_array_size))
                    tmp[:] = np.nan

                    start = 0
                    # avg_consequent_delta_widths = consequent_delta_widths.mean(axis=0)
                    avg_consequent_delta_widths = np.max(consequent_delta_widths, axis=0)
                    for q in range(self.Q):
                        end = start + self.L[q]
                        tmp[q, :self.L[q]] = avg_consequent_delta_widths[start:end]
                        start = end

                    self.term_dict['consequent_widths'] += l_rate * tmp

                self.antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes,
                                        self.antecedents, alpha=self.alpha, beta=self.beta)

                self.consequents = CLIP(batch_Y, batch_X, self.Y_mins, self.Y_maxes,
                                        self.consequents, alpha=self.alpha, beta=self.beta)

                if verbose:
                    print('Step 1: Creating/updating the fuzzy logic rules...')
                start = time.time()
                self.antecedents, self.consequents, self.rules, self.weights = rule_creation(batch_X, batch_Y,
                                                                                             self.antecedents,
                                                                                             self.consequents,
                                                                                             self.rules,
                                                                                             self.weights,
                                                                                             self.problem_type)
                K = len(self.rules)
                end = time.time()
                if verbose:
                    print('%d fuzzy logic rules created/updated in %.2f seconds.' % (K, end - start))

                if verbose:
                    consequences = [self.rules[idx]['C'][0] for idx in range(K)]
                    print('\n--- Distribution of Consequents ---')
                    print(np.unique(consequences, return_counts=True))
                    print()
                    del consequences

                self.preprocessing()

                # add or update the antecedents, consequents and rules
                if verbose:
                    print('Step 3: Creating/updating the neuro-fuzzy network...')
                start = time.time()
                self.update()
                # self.update(term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule)
                end = time.time()
                if verbose:
                    print('Neuro-fuzzy network created/updated in %.2f seconds' % (end - start))
                    print()

                start = time.time()
                rmse = self.evaluate(batch_X, batch_Y)
                end = time.time()
                if verbose:
                    print('--- Batch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse, self.K, end - start))
                    print()

                # # ANTECEDENTS

                # # adjust the array to match the self.term_dict
                # max_array_size = max(self.J.values())
                # tmp = np.empty((self.P, max_array_size))
                # tmp[:] = np.nan

                # start = 0
                # avg_antecedent_delta_c = antecedent_delta_centers.mean(axis=0)
                # for p in range(self.P):
                #     end = start + self.J[p]
                #     tmp[p, :self.J[p]] = avg_antecedent_delta_c[start:end]
                #     start = end

                # self.term_dict['antecedent_centers'] -= l_rate * tmp

                # # adjust the array to match the self.term_dict
                # max_array_size = max(self.J.values())
                # tmp = np.empty((self.P, max_array_size))
                # tmp[:] = np.nan

                # start = 0
                # avg_antecedent_delta_widths = antecedent_delta_widths.mean(axis=0)
                # for p in range(self.P):
                #     end = start + self.J[p]
                #     tmp[p, :self.J[p]] = avg_antecedent_delta_widths[start:end]
                #     start = end

                # self.term_dict['antecedent_widths'] -= l_rate * tmp

            start = time.time()
            rmse = self.evaluate(shuffled_X, shuffled_Y)
            end = time.time()
            if verbose:
                print('--- Epoch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse, self.K, end - start))
                print()

        start = time.time()
        rmse = self.evaluate(shuffled_X, shuffled_Y)
        end = time.time()
        print('--- Training RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse, self.K, end - start))
        print()

        return rmse
