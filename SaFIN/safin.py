#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:07:44 2021

@author: john
"""

import time
import random
import numpy as np

from copy import deepcopy
from functools import partial

from clip import CLIP, rule_creation
from common import boolean_indexing, RMSE, weighted_RMSE

class SaFIN:
    """
        Layer 1 consists of the input (variable) nodes.
        Layer 2 is the antecedent nodes.
        Layer 3 is the rule nodes.
        Layer 4 consists of the consequent ndoes.
        Layer 5 is the output (variable) nodes.
        
        In the SaFIN model, the input vector is denoted as:
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
    """
    def __init__(self, alpha=0.2, beta=0.6):
        """
        

        Parameters
        ----------
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.2.
        beta : TYPE, optional
            DESCRIPTION. The default is 0.6.

        Returns
        -------
        None.

        """
        self.alpha = alpha # the alpha threshold for the CLIP algorithm
        self.beta = beta # the beta threshold for the CLIP algorithm
        self.rules = [] # the fuzzy logic rules
        self.weights = [] # the weights corresponding to the rules, the i'th weight is associated with the i'th rule
        self.antecedents = []
        self.consequents = []
        self.P = None
        self.Q = None
        self.K = 0
        
    def __deepcopy__(self, memo):
        rules = deepcopy(self.rules)
        weights = deepcopy(self.weights)
        antecedents = deepcopy(self.antecedents)
        consequents = deepcopy(self.consequents)
        safin = SaFIN(deepcopy(self.alpha), deepcopy(self.beta))
        safin.load(rules, weights, antecedents, consequents)
        safin.orphaned_term_removal()
        safin.preprocessing()
        # self.K = self.consequents_indices_for_each_rule.shape[0]
        safin.update()
        return safin
    
    def load(self, rules, weights, antecedents, consequents):
        self.rules = rules
        self.weights = weights
        self.antecedents = antecedents
        self.consequents = consequents
        
        K = len(rules)
        if K > 0:
            self.K = K
            self.P = len(rules[0]['A'])
            self.Q = len(rules[0]['C'])
        
    def preprocessing(self, verbose=False):
        # make (or update) the neuro-fuzzy network
        if verbose:
            print('Step 2: Preprocessing the linguistic terms for the neuro-fuzzy network...')
        start = time.time()
        all_antecedents_centers = []
        all_antecedents_widths = []
        all_consequents_centers = []
        all_consequents_widths = []
        for p in range(self.P):
            antecedents_centers = [term['center'] for term in self.antecedents[p]]
            antecedents_widths = [term['sigma'] for term in self.antecedents[p]]
            all_antecedents_centers.append(antecedents_centers)
            all_antecedents_widths.append(antecedents_widths)
        for q in range(self.Q):
            consequents_centers = [term['center'] for term in self.consequents[q]]
            consequents_widths = [term['sigma'] for term in self.consequents[q]]
            all_consequents_centers.append(consequents_centers)
            all_consequents_widths.append(consequents_widths)
    
        self.term_dict = {}
        self.term_dict['antecedent_centers'] = boolean_indexing(all_antecedents_centers)
        self.term_dict['antecedent_widths'] = boolean_indexing(all_antecedents_widths)
        self.term_dict['consequent_centers'] = boolean_indexing(all_consequents_centers)
        self.term_dict['consequent_widths'] = boolean_indexing(all_consequents_widths)
        
        self.K = len(self.rules)
        self.antecedents_indices_for_each_rule = np.array([self.rules[k]['A'] for k in range(self.K)])
        self.consequents_indices_for_each_rule = np.array([self.rules[k]['C'] for k in range(self.K)])
        end = time.time()
        if verbose:
            print('Preprocessing completed in %.2f seconds.' % (end - start))
            print()
            
    def rule_selection(self, rule_indices_to_keep):
        # tmp = deepcopy(self.antecedents_indices_for_each_rule)
        # antecedents_indices = [list(tmp[i]) for i, antecedents_indices in enumerate(self.antecedents_indices_for_each_rule) if rule_indices_to_keep[i] == 1]
        # self.antecedents_indices_for_each_rule = np.array(antecedents_indices)
        # tmp = deepcopy(self.consequents_indices_for_each_rule)
        # consequent_indices = [list(tmp[i]) for i, consequent_indices in enumerate(self.consequents_indices_for_each_rule) if rule_indices_to_keep[i] == 1]
        # self.consequents_indices_for_each_rule = np.array(consequent_indices)
        tmp = deepcopy(self.rules)
        kept_rules = [tmp[i] for i, index in enumerate(self.rules) if rule_indices_to_keep[i] == 1]
        self.rules = kept_rules
        tmp = deepcopy(self.weights)
        kept_weights = [tmp[i] for i, index in enumerate(self.weights) if rule_indices_to_keep[i] == 1]
        self.weights = kept_weights
        self.orphaned_term_removal()
        self.preprocessing()
        # self.K = self.consequents_indices_for_each_rule.shape[0]
        error = self.update()
        if error == -1:
            self.orphaned_term_removal()
            self.preprocessing()
            self.update()
        
    def orphaned_term_removal(self):
        # need to check that no antecedent/consequent terms are "orphaned"
        
        all_antecedents = [rule['A'] for rule in self.rules]
        all_antecedents = np.array(all_antecedents)
        for p in range(self.P):
            if len(self.antecedents[p]) == len(np.unique(all_antecedents[:,p])):
                continue
            else:
                # orphaned antecedent term
                indices_for_antecedents_that_are_used = set(all_antecedents[:,p])
                updated_indices_to_map_to = list(range(len(indices_for_antecedents_that_are_used)))
                self.antecedents[p] = [self.antecedents[p][index] for index in indices_for_antecedents_that_are_used]
                
                paired_indices = list(zip(indices_for_antecedents_that_are_used, updated_indices_to_map_to))
                for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                    original_index = index_pair[0] # so, when we updated the original index to its new index
                    new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                    all_antecedents[:,p][all_antecedents[:,p] == original_index] = new_index
                
        all_consequents = [rule['C'] for rule in self.rules]
        all_consequents = np.array(all_consequents)
        for q in range(self.Q):
            if len(self.consequents[q]) == len(np.unique(all_consequents[:,q])):
                continue
            else:
                # orphaned consequent term
                indices_for_consequents_that_are_used = set(all_consequents[:,q])
                updated_indices_to_map_to = list(range(len(indices_for_consequents_that_are_used)))
                self.consequents[q] = [self.consequents[q][index] for index in indices_for_consequents_that_are_used]
                
                paired_indices = list(zip(indices_for_consequents_that_are_used, updated_indices_to_map_to))
                for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                    original_index = index_pair[0] # so, when we updated the original index to its new index
                    new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                    all_consequents[:,q][all_consequents[:,q] == original_index] = new_index
                    
        # update the rules in case any orphaned terms occurred
        for idx, rule in enumerate(self.rules):
            rule['A'] = all_antecedents[idx]
            rule['C'] = all_consequents[idx]
        
    def update(self):
    # def update(self, term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule):
    #     self.term_dict = term_dict
    #     self.antecedents_indices_for_each_rule = antecedents_indices_for_each_rule
    #     self.consequents_indices_for_each_rule = consequents_indices_for_each_rule
                
        # self.P = self.antecedents_indices_for_each_rule.shape[1]
        # self.Q = self.consequents_indices_for_each_rule.shape[1]
        # self.K = self.consequents_indices_for_each_rule.shape[0]
        
        self.J = {}
        self.total_antecedents = 0
        for p in range(self.P):
            fuzzy_clusters_in_I_p = set(self.antecedents_indices_for_each_rule[:,p])
            self.J[p] = len(fuzzy_clusters_in_I_p)
            self.total_antecedents += self.J[p]
                    
        # flat_centers = self.term_dict['antecedent_centers'].flatten()
        # flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
                
        # self.total_antecedents = len(flat_centers)
        
        # between inputs and antecedents
        self.W_1 = np.zeros((self.P, self.total_antecedents))
        start_idx = 0
        for p in range(self.P):
            end_idx = start_idx + self.J[p]
            self.W_1[p, start_idx:end_idx] = 1
            start_idx = end_idx
        
        # between antecedents and rules
        self.W_2 = np.empty((self.total_antecedents, self.K))
        self.W_2[:] = np.nan
        for rule_index, antecedents_indices_for_rule in enumerate(self.antecedents_indices_for_each_rule):
            start_idx = 0
            for input_index, antecedent_index in enumerate(antecedents_indices_for_rule):
                self.W_2[start_idx + antecedent_index, rule_index] = 1
                start_idx += self.J[input_index]
                
        self.L = {}
        self.total_consequents = 0
        for q in range(self.Q):
            fuzzy_clusters_in_O_q = set(self.consequents_indices_for_each_rule[:,q])
            self.L[q] = len(fuzzy_clusters_in_O_q)
            self.total_consequents += self.L[q]
        
        # between rules and consequents
        try:
            self.W_3 = np.empty((self.K, self.total_consequents))
            self.W_3[:] = np.nan
            for rule_index, consequent_indices_for_rule in enumerate(self.consequents_indices_for_each_rule):
                start_idx = 0
                for output_index, consequent_index in enumerate(consequent_indices_for_rule):
                    self.W_3[rule_index, start_idx + consequent_index] = 1 # IndexError: index 29 is out of bounds for axis 1 with size 29
                    start_idx += self.L[output_index]
        except IndexError:
            return -1
                
        # between consequents and outputs
        self.W_4 = np.zeros((self.total_consequents, self.Q))
        start_idx = 0
        for q in range(self.Q):
            end_idx = start_idx + self.L[q]
            self.W_4[start_idx:end_idx, q] = 1
            start_idx = end_idx
            
    def rule_pruning(self, batch_X, batch_Y, batch_size, verbose=False):
        if verbose:
            print('Step 4: Pruning unnecessary fuzzy logic rules...')
            
        start = time.time()
        
        # # multiply each rules' weight by the rules' activations
        # tmp = np.nansum(1 - self.f3, axis=0) # smaller values indicate less activation
        # tmp_1 = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) # normalize the activations to [0, 1]
        # # self.weights = list(np.mean(self.f3 * np.array(self.weights), axis=0))
        # self.weights = np.array(self.weights) * np.array(tmp_1)
        # mean = np.mean(self.weights)
        # std = np.std(self.weights)
        # # evaluation = mean / std
        # # evaluation = mean # this got decent results with alpha = 0.1, beta = 0.9
        # evaluation = np.median(self.weights)
        
        # # choices = list(np.where(np.array(self.weights) > evaluation)[0])
        # choices = list(np.where(np.array(self.weights) <= evaluation)[0])
        # if len(choices) < batch_size:
        #     NUM_TO_DELETE = len(choices)
        # else:
        #     # NUM_TO_DELETE = batch_size
        #     NUM_TO_DELETE = len(choices)
        # rule_indices = random.sample(choices, k=int(NUM_TO_DELETE / 2))
        
        from genetic_safin import objective, genetic_algorithm
        
        # define the total iterations
        n_iter = 40
        # bits
        n_bits = self.K
        # define the population size
        n_pop = 8
        # crossover rate
        r_cross = 0.9
        # mutation rate
        r_mut = 1.0 / float(n_bits)
        denominator = max(self.weights) + 0.1
        probabilities = [[(weight / denominator), 1 - (weight / denominator)] for weight in self.weights]
        best, score = genetic_algorithm(partial(objective, model=self, X=batch_X, Y=batch_Y), probabilities, 
                                        n_bits, n_iter, n_pop, r_cross, r_mut)
        print('best:')
        print(best)
        print('score: %s' % score)
        # self.rules = [rule for i, rule in enumerate(self.rules) if i in best]
        # self.weights = [weight for i, weight in enumerate(self.weights) if i in best]
        # self.rules = [rule for i, rule in enumerate(self.rules) if i not in rule_indices]
        # self.weights = [weight for i, weight in enumerate(self.weights) if i not in rule_indices]
        self.rule_selection(best)
        end = time.time()
        
        if verbose:
            print('%d fuzzy logic rules kept in %.2f seconds (original number is %d).' % (len(self.rules), end - start, len(best)))
            
    def input_layer(self, x):
        # where x is the input vector and x[i] or x_i would be the i'th element of that input vector
        # restructure the input vector into a matrix to make the condtion layer's calculations easier
        self.f1 = x
        return self.f1
    
    def condition_layer(self, o1):        
        activations = np.dot(o1, self.W_1) # the shape is (num of inputs, num of all antecedents)
        
        flat_centers = self.term_dict['antecedent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['antecedent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values

        self.f2 = np.exp(-1.0 * (np.power(activations - flat_centers, 2) / np.power(flat_widths, 2)))
        
        return self.f2 # shape is (num of inputs, num of all antecedents)
    
    def rule_base_layer(self, o2):         
        rule_activations = np.swapaxes(np.multiply(o2, self.W_2.T[:, np.newaxis]), 0, 1) # the shape is (num of observations, num of rules, num of antecedents)
        self.f3 = np.nanmin(rule_activations, axis=2) # the shape is (num of observations, num of rules)
        return self.f3
    
    def consequence_layer(self, o3):                
        consequent_activations = np.swapaxes(np.multiply(o3, self.W_3.T[:, np.newaxis]), 0, 1)
        self.f4 = np.nanmax(consequent_activations, axis=2)
        return self.f4
    
    def output_layer(self, o4):
        temp_transformation = np.swapaxes(np.multiply(o4, self.W_4.T[:, np.newaxis]), 0, 1)
        
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        
        numerator = np.nansum((temp_transformation * flat_centers * flat_widths), axis=2)
        denominator = np.nansum((temp_transformation * flat_widths), axis=2)
        self.f5 = numerator / denominator
        return self.f5

    def feedforward(self, X):
        """
        Generates output predictions for the input samples.
        
        Warning: Sensitive to the number of input samples. 
        Needs to be updated to predict using batches (i.e. may result in kernel restart).

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.

        Returns
        -------
        Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        """
        self.o1 = self.input_layer(X)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2)
        self.o4 = self.consequence_layer(self.o3)
        self.o5 = self.output_layer(self.o4)
        return self.o5
    
    def predict(self, X):
        """
        Generates output predictions for the input samples.
        
        Warning: Sensitive to the number of input samples. 
        Needs to be updated to predict using batches (i.e. may result in kernel restart).

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.

        Returns
        -------
        Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        """
        return self.feedforward(X)
    
    def evaluate(self, X, Y):
        """
        Returns the loss value & metrics values for the model in test mode.

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.
        Y : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        Returns
        -------
        RMSE : float
            The RMSE between the target and predicted Y values.

        """
        est_Y = self.predict(X)
        return RMSE(est_Y, Y), weighted_RMSE(est_Y, Y)
    
    def fit(self, X, Y, batch_size=None, epochs=1, verbose=False, shuffle=True, rule_pruning=True):
        if self.P is None:
            self.P = X.shape[1]
        if self.Q is None:
            self.Q = Y.shape[1]
        
        training_data = list(zip(X, Y))
        random.shuffle(training_data)
        shuffled_X, shuffled_Y = zip(*training_data)
        shuffled_X, shuffled_Y = np.array(shuffled_X), np.array(shuffled_Y)
        
        if batch_size is None:
            batch_size = 1
        NUM_OF_BATCHES = round(shuffled_X.shape[0] / batch_size)
        
        for epoch in range(epochs):
            for i in range(NUM_OF_BATCHES):
                print('--- Epoch %d; Batch %d ---' % (epoch + 1, i + 1))
                batch_X = X[batch_size*i:batch_size*(i+1)]
                batch_Y = Y[batch_size*i:batch_size*(i+1)]
                X_mins = np.min(batch_X, axis=0)
                X_maxes = np.max(batch_X, axis=0)
                Y_mins = np.min(batch_Y, axis=0)
                Y_maxes = np.max(batch_Y, axis=0)
                try:
                    self.X_mins = np.min([self.X_mins, X_mins], axis=0)
                    self.X_maxes = np.min([self.X_maxes, X_maxes], axis=0)
                    self.Y_mins = np.min([self.Y_mins, Y_mins], axis=0)
                    self.Y_maxes = np.min([self.Y_maxes, Y_maxes], axis=0)
                except AttributeError:
                    self.X_mins, self.X_maxes, self.Y_mins, self.Y_maxes = X_mins, X_maxes, Y_mins, Y_maxes
                    
                self.antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes, 
                                        self.antecedents, alpha=0.2, beta=0.6)
                self.consequents = CLIP(batch_Y, batch_X, Y_mins, Y_maxes, 
                                        self.consequents, alpha=self.alpha, beta=self.beta)
                
                if verbose:
                    print('Step 1: Creating/updating the fuzzy logic rules...')
                start = time.time()
                self.antecedents, self.consequents, self.rules, self.weights = rule_creation(batch_X, batch_Y, 
                                                                                             self.antecedents, 
                                                                                             self.consequents, 
                                                                                             self.rules, self.weights)
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
                rmse_before_prune, _ = self.evaluate(batch_X, batch_Y)
                end = time.time()
                print('--- Batch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
                if rule_pruning:
                    self.rule_pruning(batch_X, batch_Y, batch_size, verbose)
                print()

            start = time.time()
            rmse_before_prune, _ = self.evaluate(shuffled_X, shuffled_Y)
            end = time.time()
            print('--- Epoch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
            print()
            
        start = time.time()
        rmse_before_prune, _ = self.evaluate(shuffled_X, shuffled_Y)
        end = time.time()
        print('--- Training RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
        print()