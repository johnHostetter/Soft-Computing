#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:56:26 2021

@author: john

    This code demonstrates the Conservative Fuzzy Rule-Based Q-Learning Algorithm.

    It is inspired by the FQL code at the following link:
        https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning

    I have extended it with Tabular Conservative Q-Learning with code from:
        https://sites.google.com/view/offlinerltutorial-neurips2020/home

"""

import os
import time
import copy
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy.denfis.ecm import ECM
from rl.cql import CQLModel
from fuzzy.neuro.adaptive import AdaptiveNeuroFuzzy
from fuzzy.self_adaptive.clip import CLIP, rule_creation

GLOBAL_SEED = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(GLOBAL_SEED)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(GLOBAL_SEED)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(GLOBAL_SEED)

# 4. Set the `torch` pseudo-random generator at a fixed value
torch.manual_seed(GLOBAL_SEED)

class CFQLModel(AdaptiveNeuroFuzzy, CQLModel):
    def __init__(self, clip_params, fis_params, cql_params):
        # super().__init__()
        AdaptiveNeuroFuzzy.__init__(self)
        CQLModel.__init__(self, cql_params)

        self.current_rule_activations = []

        # the alpha parameter for the CLIP algorithm
        self.alpha = clip_params['alpha']
        # the beta parameter for the CLIP algorithm
        self.beta = clip_params['beta']

        # the inference engine to use for fuzzy logic control
        self.inference_engine = fis_params['inference_engine']

    def d(self, x):
        return self.truth_value(x).sum()

    def infer(self, x):
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
        self.truth_value(x)
        numerator = (self.current_rule_activations[:, np.newaxis] * self.q_table).sum(axis=0)
        denominator = self.current_rule_activations.sum()
        q_values = numerator / denominator
        return q_values

    def c_k(self, train_X, k):
        # we just need the rule activations in layer 3, ignore the returned values
        self.predict(train_X)
        lhs = 1 / self.f3.sum(axis=1)
        rhs = abs(self.q_table - self.q_table[k]).max()
        return lhs * rhs

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
        self.current_rule_activations = copy.deepcopy(self.o3[0])
        return self.current_rule_activations

    def get_action(self, state):
        """
        Given the provided state value, returns the index of the possible actions that should be taken.

        Parameters
        ----------
        state : 2-D Numpy array
            The state that the rules' degrees of applicability should be calculated for to facilitate fuzzy inference.

        Returns
        -------
        int
            The index of the action that should be taken.

        """
        q_values = self.infer(state)
        return np.argmax(q_values)

    def export_antecedents(self):
        """
        Export the antecedent terms for each input variable of the Neuro-Fuzzy network into a Pandas DataFrame representation.

        Returns
        -------
        Pandas DataFrame
            The antecedent terms for each input variable.

        """
        all_antecedents = []
        for input_idx, antecedents_for_input in enumerate(self.antecedents):
            for term_idx, antecedent in enumerate(antecedents_for_input):
                antecedent['input_variable'] = input_idx
                antecedent['term_index'] = term_idx
                all_antecedents.append(antecedent)
        return pd.DataFrame(all_antecedents)

    def export_consequents(self):
        """
        Export the consequent terms for each output variable of the Neuro-Fuzzy network into a Pandas DataFrame representation.

        Returns
        -------
        Pandas DataFrame
            The consequent terms for each output variable.

        """
        all_consequents = []
        for input_idx, consequents_for_input in enumerate(self.consequents):
            for term_idx, consequent in enumerate(consequents_for_input):
                consequent['output_variable'] = input_idx
                consequent['term_index'] = term_idx
                all_consequents.append(consequent)
        return pd.DataFrame(all_consequents)

    def export_q_values(self):
        """
        Export the Q-values for each fuzzy logic rule of the Neuro-Fuzzy network into a Pandas DataFrame representation.
        The i'th row corresponds to the i'th fuzzy logic rule in the Neuro-Fuzzy network (order matters).
        The j'th column corresponds to the j'th possible action's Q-value.

        Returns
        -------
        Pandas DataFrame
            The Q-values for each fuzzy logic rule's possible actions.

        """
        return pd.DataFrame(self.q_table)

    def export_rules(self):
        """
        Export the fuzzy logic rules of the Neuro-Fuzzy network into a Pandas DataFrame representation.

        Returns
        -------
        Pandas DataFrame
            The fuzzy logic rules.

        """
        return pd.DataFrame(self.rules)

    def save(self, file_name):
        """
        Save this Neuro-Fuzzy network for future use. The output is '.csv' files.

        Parameters
        ----------
        file_name : TYPE
            The name of a file or file path to use in saving the Neuro-Fuzzy network.
            Should not include the '_q_values.csv', '_rules.csv', '_antecedents.csv', etc. extensions.
            These are automatically appended by the 'save' function.

        Returns
        -------
        None.

        """
        antecedents_df = self.export_antecedents()
        consequents_df = self.export_consequents()
        q_values_df = self.export_q_values()
        rules_df = self.export_rules()
        antecedents_df.to_csv('{}_antecedents.csv'.format(file_name), sep=',', index=False)
        consequents_df.to_csv('{}_consequents.csv'.format(file_name), sep=',', index=False)
        q_values_df.to_csv('{}_q_values.csv'.format(file_name), sep=',', index=False)
        rules_df.to_csv('{}_rules.csv'.format(file_name), sep=',', index=False)

    def load(self, file_name):
        """
        Load an existing Neuro-Fuzzy network that has been trained or untrained.

        Parameters
        ----------
        file_name : string
            The name of a file or file path to use in locating the saved Neuro-Fuzzy network.
            Should not include the '_q_values.csv', '_rules.csv', '_antecedents.csv', etc. extensions.
            These are automatically appended by the 'save' function.

        Returns
        -------
        None.

        """
        self.q_table = pd.read_csv('{}_q_values.csv'.format(file_name)).values
        self.rules = pd.read_csv(
            '{}_rules.csv'.format(file_name)).to_dict('records')

        # convert the rules' antecedents and consequents references from the string representation to the Numpy format
        for rule in self.rules:
            antecedents_indices_list = rule['A'][1:-1].split(' ')
            antecedents_indices_list = np.array(
                [int(index) for index in antecedents_indices_list])
            rule['A'] = antecedents_indices_list
            consequents_indices_list = rule['C'][1:-1].split(' ')
            consequents_indices_list = np.array(
                [int(index) for index in consequents_indices_list])
            rule['C'] = consequents_indices_list

        antecedents = pd.read_csv(
            '{}_antecedents.csv'.format(file_name)).to_dict('records')
        # the loaded antecedents need to be reformatted so they are first indexed by their input variable's index
        self.antecedents = []
        current_input_variable_index = 0
        current_input_variable_antecedents = []
        for antecedent in antecedents:
            if antecedent['input_variable'] == current_input_variable_index:
                current_input_variable_antecedents.append(antecedent)
            else:
                current_input_variable_index = antecedent['input_variable']
                self.antecedents.append(current_input_variable_antecedents)
                current_input_variable_antecedents = [antecedent]
        # add the last created current_input_variable_antecedents
        current_input_variable_index = antecedent['input_variable']
        self.antecedents.append(current_input_variable_antecedents)
        current_input_variable_antecedents = []

        # for simplicity, the consequents are loaded again, however, they are not used for inference, but are used to generate the Neuro-Fuzzy network
        consequents = pd.read_csv(
            '{}_consequents.csv'.format(file_name)).to_dict('records')
        # the loaded consequents need to be reformatted so they are first indexed by their output variable's index
        self.consequents = []
        current_output_variable_index = 0
        current_output_variable_consequents = []
        for consequent in consequents:
            if consequent['output_variable'] == current_output_variable_index:
                current_output_variable_consequents.append(consequent)
            else:
                current_output_variable_index = consequent['output_variable']
                self.consequents.append(current_output_variable_consequents)
                current_output_variable_consequents = [consequent]
        # add the last created current_output_variable_consequents
        current_output_variable_index = consequent['output_variable']
        self.consequents.append(current_output_variable_consequents)
        current_output_variable_consequents = []

        # currently, the weights of each rule are not saved since they do not matter at this point
        self.weights = np.ones(len(self.rules))
        # building the Neuro-Fuzzy network
        self.import_existing(self.rules, self.weights,
                             self.antecedents, self.consequents)
        self.orphaned_term_removal()
        self.preprocessing()
        self.update()

    def fit(self, train_X, trajectories, ecm=False, Dthr=1e-3, prune_rules=False,
            apfrb_sensitivity_analysis=False, verbose=False):
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
        prune_rules : boolean, optional
            This boolean controls whether to further prune the candidate rules. The rule pruning strategy is removing all
            rules that are activated less than the average rule degree activation. If the 'ECM' algorithm and 'prune_rules'
            are both False, then the candidate rule generation procedure is the Wang and Mendel approach. The default is False.
        apfrb_sensitivity_analysis : boolean, optional
            Enables a post-hoc sensitivity analysis of the fuzzy logic rules using the procedure introduced in the APFRB paper.
            The default is False.
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

        start = time.time()
        self.antecedents = CLIP(train_X, dummy_Y, train_X_mins, train_X_maxes,
                                self.antecedents, alpha=self.alpha, beta=self.beta)
        end = time.time()
        if verbose:
            print('membership functions for the antecedents generated in %.2f seconds.' % (
                end - start))

        start = time.time()
        self.consequents = CLIP(dummy_Y, train_X, Y_mins, Y_maxes, self.consequents,
                                alpha=self.alpha, beta=self.beta)
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
        self.antecedents, self.consequents, self.rules, self.weights = rule_creation(reduced_X, reduced_dummy_Y,
                                                                                     self.antecedents,
                                                                                     self.consequents,
                                                                                     self.rules,
                                                                                     self.weights,
                                                                                     problem_type='SL',
                                                                                     consistency_check=False)

        K = len(self.rules)
        end = time.time()
        if verbose:
            print('%d fuzzy logic rules created/updated in %.2f seconds.' %
                  (K, end - start))

        if verbose:
            print('\nBuilding the initial Neuro-Fuzzy Network...')
        start = time.time()
        self.import_existing(self.rules, self.weights,
                             self.antecedents, self.consequents)
        self.orphaned_term_removal()
        self.preprocessing()
        self.update()
        end = time.time()
        if verbose:
            print('done; built in %.2f seconds.' % (end - start))

        if prune_rules:
            if verbose:
                print(
                    '\nDetermining the average activation of each fuzzy logic rule on the training data...')
            start = time.time()
            o1 = self.input_layer(train_X)
            o2 = self.condition_layer(o1)
            o3 = self.rule_base_layer(o2, inference=self.inference_engine)
            mean_rule_activations = o3.mean(axis=0)
            end = time.time()
            if verbose:
                print('done; calculated in %.2f seconds.' % (end - start))
            plt.bar([x for x in range(len(mean_rule_activations))],
                    mean_rule_activations)
            plt.show()

            if verbose:
                print('\nRemoving weakly activated fuzzy logic rules...')
            indices = np.where(mean_rule_activations >
                               np.mean(mean_rule_activations))[0]
            self.rules = [self.rules[index] for index in indices]
            self.weights = [self.weights[index] for index in indices]
            self.import_existing(self.rules, self.weights,
                                 self.antecedents, self.consequents)
            self.orphaned_term_removal()
            self.preprocessing()
            self.update()
            if verbose:
                print(
                    'done; removal of fuzzy logic rules has been reflected in the Neuro-Fuzzy network.')

        # prepare the Q-table
        self.build_q_table(self.get_number_of_rules())

        if verbose:
            print('\nTransforming the trajectories to a compatible format...')
        start = time.time()
        rule_weights = []
        transformed_trajectories = []
        for trajectory in trajectories:
            state = trajectory[0]
            action_index = trajectory[1]
            reward = trajectory[2]
            next_state = trajectory[3]
            done = trajectory[4]

            # this code will only use the rule with the highest degree of activation to do updates
            self.truth_value(state)
            rule_index = np.argmax(self.current_rule_activations)
            rule_index_weight = self.current_rule_activations[rule_index]
            rule_weights.append(rule_index_weight)
            self.truth_value(next_state)
            next_rule_index = np.argmax(self.current_rule_activations)
            next_rule_index_weight = self.current_rule_activations[next_rule_index]
            transformed_trajectories.append(
                (rule_index, action_index, reward, next_rule_index, done))
        end = time.time()
        if verbose:
            print('done; trajectories transformed in %.2f seconds.' %
                  (end - start))

        if verbose:
            print('\nBegin Offline [Tabular] Conservative Q-Learning...')
        start = time.time()
        self.conservative_q_iteration(sampled=True, training_dataset=transformed_trajectories, rule_weights=rule_weights)
        end = time.time()
        if verbose:
            print('done; completed in %.2f seconds.' % (end - start))

        # APFRB sensitivity analysis
        if apfrb_sensitivity_analysis:
            if verbose:
                print('\nPerforming sensitivity analysis inspired by APFRB paper...')
            start = time.time()
            l_ks = []
            for k, _ in enumerate(self.rules):
                print(k)
                max_c_k = np.max(self.c_k(train_X, k))
                l_ks.append(max_c_k)
            end = time.time()
            if verbose:
                print('done; completed in %.2f seconds.' % (end - start))

            plt.bar([x for x in range(len(l_ks))], l_ks)
            plt.show()

            m_k_c_k = np.median(self.f3, axis=0) * l_ks

            plt.bar([x for x in range(len(m_k_c_k))], m_k_c_k)
            plt.show()

        # disable learning and exploration

        self.gamma = 0.0
        self.alpha = 0.0
