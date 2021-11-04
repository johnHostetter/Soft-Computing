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

from nfn import ModifyRulesNeuroFuzzy
from clip import CLIP, rule_creation

# SaFIN technically can only use AdaptiveNeuroFuzzy, but I use ModifyRulesNeuroFuzzy instead to test rule pruning functionality on Boston dataset
class SaFIN(ModifyRulesNeuroFuzzy):
    """
        Layer 1 consists of the input (variable) nodes.
        Layer 2 is the antecedent nodes.
        Layer 3 is the rule nodes.
        Layer 4 consists of the consequent nodes.
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
    def __init__(self, alpha=0.2, beta=0.6, X_mins=None, X_maxes=None):
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
        super().__init__()
        self.alpha = alpha # the alpha threshold for the CLIP algorithm
        self.beta = beta # the beta threshold for the CLIP algorithm
        if X_mins is not None and X_maxes is not None:
            self.X_mins = X_mins
            self.X_maxes = X_maxes
        else:
            self.X_mins = None
            self.X_maxes = None
        
    def __deepcopy__(self, memo):
        rules = deepcopy(self.rules)
        weights = deepcopy(self.weights)
        antecedents = deepcopy(self.antecedents)
        consequents = deepcopy(self.consequents)
        safin = SaFIN(deepcopy(self.alpha), deepcopy(self.beta))
        safin.import_existing(rules, weights, antecedents, consequents)
        safin.X_mins = deepcopy(self.X_mins)
        safin.X_maxes = deepcopy(self.X_maxes)
        if self.P is not None:
            safin.orphaned_term_removal()
            safin.preprocessing()
            safin.update()
        return safin
    
    def backpropagation(self, x, y):
        # (1) calculating the error signal in the output layer
        
        e5_m = y - self.o5 # y actual minus y predicted
        e5 = np.dot(e5_m, self.W_4.T) # assign the error to its corresponding output node
        error = (self.o4 * e5)
        
        # delta centers
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        # y4_k = (centers * self.W_4.T)
        # numerator = (widths * y4_k)
        widths = np.multiply(flat_widths[:,np.newaxis], self.W_4).T
        num = np.multiply(widths[np.newaxis,:,:], self.o4[:, np.newaxis,:])
        den = num.sum(axis=2)
        consequent_delta_c = (num / den[:, :, np.newaxis]).sum(axis=1)
        
        # delta widths
        c_lk = (flat_centers * self.W_4.T)
        lhs_term = np.dot(den, c_lk)
        rhs_term = np.multiply(num, c_lk)
        compatible_rhs_term = rhs_term.sum(axis=1)
        difference = lhs_term - compatible_rhs_term
        numerator = np.multiply(self.o4, difference)
        denominator = np.power(den, 2)
        compatible_numerator = np.multiply(numerator[:,np.newaxis], self.W_4.T)
        division = (compatible_numerator / denominator[:, :, np.newaxis])
        consequent_delta_widths = division.sum(axis=1)
        
        # layer 4 error signal
        numerator = (flat_widths * difference)
        compatible_numerator = np.multiply(numerator[:,np.newaxis], self.W_4.T)
        division = (compatible_numerator / denominator[:, :, np.newaxis])
        layer_4_error_rhs = division.sum(axis=1)
        layer_4_error = error * layer_4_error_rhs
        
        # layer 3 error signal
        layer_3_error = (layer_4_error[:,np.newaxis,:] * self.W_3)
        layer_3_error = np.nansum(layer_3_error, axis=2)
        
        # layer 2 error signal
        antecedent_activations = (self.o2[:,np.newaxis,:] * self.W_2.T) # shape is (num of observations, num of rules, num of antecedents)
        y2_i = (self.o2[:,np.newaxis,:] * antecedent_activations)
        r = np.nanargmin(y2_i, axis=2) # shape is (num of observations, num of rules)
        dE_dy_i = np.zeros(self.o2.shape)
        
        for observation_idx in range(r.shape[0]):
            antecedent_indices = np.unique(r[observation_idx])
            for antecedent_index in antecedent_indices:
                rule_indices = np.where(r[observation_idx] == antecedent_index)[0]
                dE_dy_i[observation_idx, antecedent_index] = layer_3_error[observation_idx, rule_indices].sum()
                
        # delta centers
        antecedent_delta_c_lhs = (np.multiply(dE_dy_i, self.o2))
        # remove the stored np.nan values
        shape = antecedent_delta_c_lhs.shape
        antecedent_delta_c_lhs = antecedent_delta_c_lhs[~np.isnan(antecedent_delta_c_lhs)]
        antecedent_delta_c_lhs = antecedent_delta_c_lhs.reshape(shape[0], int(antecedent_delta_c_lhs.shape[0] / shape[0])) # shape is (num of observations, num of all antecedents [nonapplicable antecedents removed])
        antecedent_delta_c_rhs_num = 2 * (x[:,:,np.newaxis] - self.term_dict['antecedent_centers'])
        antecedent_delta_c_rhs_den = np.power(self.term_dict['antecedent_widths'], 2)
        antecedent_delta_c_rhs_den = np.where(antecedent_delta_c_rhs_den == 0.0, np.finfo(np.float).eps, antecedent_delta_c_rhs_den) # if there is a zero in the denominator, replace it with the smallest possible float value, otherwise, keep the other values

        antecedent_delta_c_rhs = (antecedent_delta_c_rhs_num / antecedent_delta_c_rhs_den)
        shape = antecedent_delta_c_rhs.shape
        compatible_antecedent_delta_c_rhs = antecedent_delta_c_rhs.reshape(shape[0], shape[1]*shape[2]) # shape[0] is num of observations, shape[1] is num of input nodes, shape[2] is maximum number of linguistic terms possible
        # remove the stored np.nan values
        compatible_antecedent_delta_c_rhs = compatible_antecedent_delta_c_rhs[~np.isnan(compatible_antecedent_delta_c_rhs)].reshape(antecedent_delta_c_lhs.shape)
        antecedent_delta_c = antecedent_delta_c_lhs * compatible_antecedent_delta_c_rhs
        
        # delta widths
        antecedent_delta_widths_rhs_num = 2 * np.power((x[:,:,np.newaxis] - self.term_dict['antecedent_centers']), 2)
        antecedent_delta_widths_rhs_den = np.power(self.term_dict['antecedent_widths'], 3)
        antecedent_delta_widths_rhs_den = np.where(antecedent_delta_widths_rhs_den == 0.0, np.finfo(np.float).eps, antecedent_delta_widths_rhs_den) # if there is a zero in the denominator, replace it with the smallest possible float value, otherwise, keep the other values
        antecedent_delta_widths_rhs = (antecedent_delta_widths_rhs_num / antecedent_delta_widths_rhs_den)
        shape = antecedent_delta_widths_rhs.shape
        compatible_antecedent_delta_widths_rhs = antecedent_delta_widths_rhs.reshape(shape[0], shape[1]*shape[2]) # shape[0] is num of observations, shape[1] is num of input nodes, shape[2] is maximum number of linguistic terms possible
        # remove the stored np.nan values
        compatible_antecedent_delta_widths_rhs = compatible_antecedent_delta_widths_rhs[~np.isnan(compatible_antecedent_delta_widths_rhs)].reshape(antecedent_delta_c_lhs.shape)
        
        antecedent_delta_widths = antecedent_delta_c_lhs * compatible_antecedent_delta_widths_rhs
        
        return consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths
    
    def fit(self, X, Y, batch_size=None, epochs=1, verbose=False, shuffle=True, rule_pruning=True, problem_type='SL'):
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
                    
                self.antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes, 
                                        self.antecedents, alpha=self.alpha, beta=self.beta)
                
                if problem_type == 'SL':
                    self.consequents = CLIP(batch_Y, batch_X, Y_mins, Y_maxes, 
                                            self.consequents, alpha=self.alpha, beta=self.beta)
                elif problem_type == 'RL':
                    if len(self.consequents) == 0:
                        for q in range(self.Q):
                            self.consequents.append([])
                
                if verbose:
                    print('Step 1: Creating/updating the fuzzy logic rules...')
                start = time.time()
                self.antecedents, self.consequents, self.rules, self.weights = rule_creation(batch_X, batch_Y, 
                                                                                             self.antecedents, 
                                                                                             self.consequents, 
                                                                                             self.rules, 
                                                                                             self.weights,
                                                                                             problem_type)
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
                rmse_before_prune = self.evaluate(batch_X, batch_Y)
                end = time.time()
                print('--- Batch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
                if rule_pruning:
                    self.rule_pruning(batch_X, batch_Y, batch_size, verbose)
                    rmse_after_prune = self.evaluate(batch_X, batch_Y) # we need to update the stored values e.g. self.o4
                print()
                
                l_rate = 0.05
                # l_rate = 0.001
                consequent_delta_c, consequent_delta_widths, antecedent_delta_centers, antecedent_delta_widths = self.backpropagation(batch_X, batch_Y)
                
                # self.term_dict['consequent_centers'] -= l_rate * np.reshape(consequent_delta_c.mean(axis=0), self.term_dict['consequent_centers'].shape)
                # adjust the array to match the self.term_dict
                max_array_size = max(self.L.values())
                tmp = np.empty((self.Q, max_array_size))
                tmp[:] = np.nan
                
                start = 0
                avg_consequent_delta_c = consequent_delta_c.mean(axis=0)
                for q in range(self.Q):
                    end = start + self.L[q]
                    tmp[q, :self.L[q]] = avg_consequent_delta_c[start:end]
                    start = end                    
                    
                self.term_dict['consequent_centers'] -= l_rate * tmp
                    
                # adjust the array to match the self.term_dict
                max_array_size = max(self.L.values())
                tmp = np.empty((self.Q, max_array_size))
                tmp[:] = np.nan
                
                start = 0
                avg_consequent_delta_widths = consequent_delta_widths.mean(axis=0)
                for q in range(self.Q):
                    end = start + self.L[q]
                    tmp[q, :self.L[q]] = avg_consequent_delta_widths[start:end]
                    start = end                    
                    
                self.term_dict['consequent_widths'] -= l_rate * tmp

                # ANTECEDENTS
                
                # adjust the array to match the self.term_dict
                max_array_size = max(self.J.values())
                tmp = np.empty((self.P, max_array_size))
                tmp[:] = np.nan
                
                start = 0
                avg_antecedent_delta_c = antecedent_delta_centers.mean(axis=0)
                for p in range(self.P):
                    end = start + self.J[p]
                    tmp[p, :self.J[p]] = avg_antecedent_delta_c[start:end]
                    start = end                    
                    
                self.term_dict['antecedent_centers'] -= l_rate * tmp

                # adjust the array to match the self.term_dict
                max_array_size = max(self.J.values())
                tmp = np.empty((self.P, max_array_size))
                tmp[:] = np.nan
                
                start = 0
                avg_antecedent_delta_widths = antecedent_delta_widths.mean(axis=0)
                for p in range(self.P):
                    end = start + self.J[p]
                    tmp[p, :self.J[p]] = avg_antecedent_delta_widths[start:end]
                    start = end                    
                    
                self.term_dict['antecedent_widths'] -= l_rate * tmp
                
            start = time.time()
            rmse_before_prune = self.evaluate(shuffled_X, shuffled_Y)
            end = time.time()
            print('--- Epoch RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
            print()
            
        start = time.time()
        rmse_before_prune = self.evaluate(shuffled_X, shuffled_Y)
        end = time.time()
        print('--- Training RMSE = %.6f with %d Number of Fuzzy Logic Rules in %.2f seconds ---' % (rmse_before_prune, self.K, end - start))
        print()
        
        return rmse_before_prune