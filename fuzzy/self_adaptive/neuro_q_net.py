#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:07:51 2022

@author: john
"""

# demo of error back-propagation training algorithm

import torch
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from fuzzy.neuro.adaptive import AdaptiveNeuroFuzzy


# Yield successive n-sized chunks from l.
def divide_chunks(lst, n):
    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class FLC(AdaptiveNeuroFuzzy):
    def __init__(self, antecedents, rules, consequents, consequent_term, cql_alpha, learning_rate=1e-3):
        # the consequent term is a fuzzy set that is to be ignored, it is only for construction of neuro-fuzzy net
        AdaptiveNeuroFuzzy.__init__(self)

        # self.x = {} # antecedent term's center (Gaussian)
        # self.sigma = {} # antecedent terms' width (Gaussian)
        self.antecedents = antecedents
        # self.rules = rules  # dictionary to map the l'th rule to a tuple of n antecedents' terms
        self.legacy_rules = [tuple(rule['A']) for rule in rules]
        self.y = consequents  # l'th rule's consequent center
        self.M = len(self.legacy_rules)
        self.learning_rate = learning_rate
        self.b_memo = {}  # memoization
        self.z_memo = {}  # memoization
        self.cql_alpha = cql_alpha

        # for speed, build the neuro-fuzzy network
        self.inference_engine = 'product'
        self.import_existing(rules, [1.0] * len(self.legacy_rules),
                             self.antecedents, consequent_term)
        self.orphaned_term_removal()
        self.preprocessing()
        super().update()

    # Fuzzify to get the degree of truth values
    def truth_value(self, state_value):
        """
        Calculates the degree of applicability of each fuzzy logic rule, using the Neuro-Fuzzy network.

        Parameters
        ----------
        state_value : 2-D Numpy array
            The state that the rules' degrees of applicability should be calculated for to facilitate fuzzy inference.

        Returns
        -------
        Numpy 2-D array
            The degree of applicability of each fuzzy logic rule in the third layer,
            has a shape of (number of observations, number of rules).

        """
        self.o1 = self.input_layer(state_value)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2, inference=self.inference_engine)
        self.current_rule_activations = self.o3
        return self.current_rule_activations

    def z(self, l):
        # l'th fuzzy logic rule's activation
        return self.current_rule_activations[:, l]

    def a(self):
        # numerator
        q_table = np.array(self.y)[:, np.newaxis]  # shape is now (num of rules, num of actions)
        return self.current_rule_activations.dot(q_table)

    def b(self):
        # denominator
        return self.current_rule_activations.sum(axis=1)

    def f(self, x):
        # fuzzy inference
        self.truth_value(x)  # x can be multiple observations
        q_values = self.a() / self.b()[:, np.newaxis]
        return torch.Tensor(q_values)

    def predict(self, x):
        return self.f(x)

    def legacy_z(self, l, x):
        # l'th fuzzy logic rule's activation
        value = 1.0
        rule = self.legacy_rules[l]
        n = len(rule)

        key = [l]
        key.extend(x.tolist())
        key = tuple(key)
        if key in self.z_memo:
            return self.z_memo[key]
        else:
            for i in range(n):
                term_idx = rule[i]
                term = self.antecedents[i][term_idx]
                quotient = (x[i] - term['center']) / term['sigma']
                u = np.power(quotient, 2)
                value *= np.exp(-1 * u)
            self.z_memo[key] = value
            return value

    def legacy_a(self, x):
        # numerator
        value = 0.0
        for l in range(self.M):
            value += self.y[l] * self.legacy_z(l, x)
            # value += self.y[l] * self.current_rule_activations[l]
        return value

    def legacy_b(self, x):
        # denominator
        key = tuple(x.tolist())
        if key in self.b_memo:
            return self.b_memo[key]
        else:
            value = 0.0
            for l in range(self.M):
                value += self.legacy_z(l, x)
                # value += self.current_rule_activations[l]
            self.b_memo[key] = value
            return value

    def legacy_predict(self, x):
        # fuzzy inference
        return self.legacy_a(x) / self.legacy_b(x)


class AdamOptim():
    # https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, dw, b=None, db=None):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        if b is not None and db is not None:
            self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # *** biases *** #
        if b is not None and db is not None:
            self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db)

        # bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        if b is not None and db is not None:
            m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        if b is not None and db is not None:
            v_db_corr = self.v_db / (1 - self.beta2 ** t)

        # update weights and biases
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        if b is not None and db is not None:
            b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
        else:
            b = np.nan
        return w, b


class MIMO:
    def __init__(self, normalizer, antecedents, rules, n_outputs, consequent_term, cql_alpha, learning_rate=1e-3):
        self.timestep = 0
        self.cql_alpha = cql_alpha
        self.normalizer = normalizer
        self.learning_rate = learning_rate
        self.adam = AdamOptim(eta=learning_rate)
        self.flcs = []

        for _ in range(n_outputs):
            consequents = np.zeros(len(rules))
            # consequents = np.random.uniform(0, 1.0, len(rules))
            self.flcs.append(FLC(antecedents, rules, consequents, consequent_term, cql_alpha, learning_rate))

    def predict(self, x):
        """
        A custom fuzzy inference procedure, that uses the current rule activations
        to weigh their corresponding rule's Q-values.

        Parameters
        ----------
        x : 2-D Numpy array
            DESCRIPTION.

        Returns
        -------
        q_values : 1-D Numpy array
            An array of the actions' Q-values, where the i'th Q-value corresponds to the i'th possible action.

        """
        x = self.normalizer.transform(x)
        self.flcs[0].truth_value(x)  # x can be multiple observations
        q_table = []
        for flc in self.flcs:
            q_table.append(flc.y)
        q_table = np.array(q_table).T  # shape is now (num of rules, num of actions)
        numerator = self.flcs[0].current_rule_activations.dot(q_table)
        denominator = self.flcs[0].current_rule_activations.sum(axis=1)
        q_values = numerator / denominator[:, np.newaxis]
        return torch.Tensor(q_values)

    def calc_flc_outputs(self, X):
        flc_outputs = []
        for flc_idx, flc in enumerate(self.flcs):
            flc_outputs.append([])
            for training_idx, x in enumerate(X):
                flc_outputs[flc_idx].append(flc.legacy_predict(x))

        return np.array(flc_outputs)

    def calc_offline_loss(self, flc_outputs, y, action_indices):
        target_qvalues = torch.tensor(y, dtype=torch.float32)
        action_indices = torch.tensor(action_indices, dtype=torch.int64)
        pred_qvalues = torch.tensor(flc_outputs, dtype=torch.float32)
        logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)

        tmp_pred_qvalues = pred_qvalues.gather(
            1, action_indices.reshape(-1, 1)).squeeze()
        cql_loss = logsumexp_qvalues - tmp_pred_qvalues

        loss = torch.mean((pred_qvalues - target_qvalues) ** 2)
        loss = loss + self.cql_alpha * torch.mean(cql_loss)
        return loss

    def offline_update(self, train_X, train_y, train_action_indices, val_X, val_y, val_action_indices):
        # calculate the outputs

        train_flc_outputs = self.predict(train_X).numpy()
        val_flc_outputs = self.predict(val_X).numpy()

        # calculate the loss for reporting & monitoring

        train_loss = self.calc_offline_loss(train_flc_outputs, train_y, train_action_indices)
        val_loss = self.calc_offline_loss(val_flc_outputs, val_y, val_action_indices)

        # gradient descent

        for flc_idx, flc in enumerate(self.flcs):
            if train_y.ndim == 1:  # single observation
                y_ = train_y[flc_idx]
            elif train_y.ndim == 2:  # multiple observations
                y_ = train_y[:, flc_idx]

            constant = 2 / len(y_)
            loss_on_train_data = (flc.predict(train_X) - y_[:, np.newaxis])
            mse = np.zeros(self.flcs[0].M)
            offlines = np.zeros(self.flcs[0].M)

            for l in range(flc.M):
                z = flc.z(l)
                # consequent terms' centers
                b = flc.b()
                # mse[l] = float(constant * (loss_on_train_data * (z / b)[:, np.newaxis]))
                mse[l] = float((loss_on_train_data.T[0] * (z / b)).sum() * constant)

                # correct offline
                numerator = (np.exp(train_flc_outputs[:, flc_idx]) * z / b)
                denominator = (np.log(2) * np.exp(train_flc_outputs)).sum(axis=1)
                offlines[l] = (((numerator / denominator) - (z / b)).mean())

            offline_adjustment = self.cql_alpha * offlines
            # learning_rate = 6.25e-05
            flc.y, _ = self.adam.update(self.timestep, w=flc.y, dw=(offline_adjustment + mse))
            # flc.y -= learning_rate * (offline_adjustment + mse)
        return train_loss, val_loss


class MIMO_replay(MIMO):
    def extract_data_from_trajectories(self, batch, gamma):
        batch = np.array(batch)
        states = np.array([list(state) for state in batch[:, 0]])
        next_states = np.array([list(next_state) for next_state in batch[:, 2]])
        action_indices = batch[:, 1].astype('int32')
        q_values = self.predict(states)
        q_values_next = self.predict(next_states)
        rewards = batch[:, 3].astype('float32')
        dones = batch[:, 4]
        q_values[torch.arange(len(q_values)), action_indices] = torch.tensor(rewards)
        max_q_values_next, max_q_values_next_indices = torch.max(q_values_next, dim=1)
        max_q_values_next *= gamma
        q_values[torch.arange(len(q_values)), action_indices] += torch.tensor(
            [0.0 if done else float(max_q_values_next[idx]) for idx, done in enumerate(dones)])
        targets = q_values.numpy()

        return states, targets, action_indices

    def replay(self, train_memory, size, validation_memory, gamma=0.9, online=True):
        """ Add experience replay to the DQN network class. """
        # Make sure the memory is big enough
        if len(train_memory) >= size:
            # Sample a batch of experiences from the agent's memory
            # batch = random.sample(train_memory, size)

            # divide the memory into batches of length 'size'
            batches = list(divide_chunks(train_memory, size))

            train_losses = []
            val_losses = []
            for batch in batches:
                self.timestep += 1  # the number of gradient descent updates performed thus far

                # Extract information from the data
                train_states, train_targets, train_action_indices = self.extract_data_from_trajectories(batch, gamma)

                val_states, val_targets, val_action_indices = self.extract_data_from_trajectories(validation_memory,
                                                                                                  gamma)

                if online:  # needs updating (to remove validation data arguments)
                    train_loss, val_loss = self.offline_update(train_states, train_targets, train_action_indices,
                                                               val_states, val_targets, val_action_indices)
                else:
                    train_loss, val_loss = self.offline_update(train_states, train_targets, train_action_indices,
                                                               val_states, val_targets, val_action_indices)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            return np.array(train_losses), np.array(val_losses)


class EvaluationWrapper:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, state):
        return [torch.argmax(self.agent.predict(state)).item()]


class DoubleMIMO(MIMO):
    def __init__(self, antecedents, rules, n_outputs, consequent_term, cql_alpha, learning_rate=1e-3):
        super().__init__(antecedents, rules, n_outputs, consequent_term, cql_alpha, learning_rate)
        self.target = MIMO(antecedents, rules, n_outputs, consequent_term, cql_alpha, learning_rate)

    def target_predict(self, state):
        """ Use target network to make predictions. """
        with torch.no_grad():
            return self.target.predict(state)

    def target_update(self):
        """ Update target network with the model weights. """
        for idx, target_flc in enumerate(self.target.flcs):
            target_flc.consequents = self.flcs[idx].consequents

    def replay(self, memory, size, gamma=0.9, online=True):
        """ Add experience replay to the DQL network class. """
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract data points from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict([state]).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict([next_state])
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            if online:
                self.update(np.array(states), np.array(targets))
            else:
                self.offline_update(np.array(states), np.array(targets))
