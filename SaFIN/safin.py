#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:07:44 2021

@author: john
"""

import numpy as np

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
    def __init__(self, term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule):
        """
        
        Parameters
        ----------
        term_dict : TYPE
            DESCRIPTION.
        antecedents_indices_for_each_rule : TYPE
            DESCRIPTION.
        consequents_indices_for_each_rule : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.term_dict = term_dict
        self.antecedents_indices_for_each_rule = antecedents_indices_for_each_rule
        self.consequents_indices_for_each_rule = consequents_indices_for_each_rule
                
        self.P = self.antecedents_indices_for_each_rule.shape[1]
        # temporary fix until MIMO
        self.consequents_indices_for_each_rule = np.reshape(consequents_indices_for_each_rule, (len(consequents_indices_for_each_rule), 1))
        self.Q = self.consequents_indices_for_each_rule.shape[1]
        self.K = self.consequents_indices_for_each_rule.shape[0]
        
        self.J = {}
        self.total_antecedents = 0
        for p in range(self.P):
            fuzzy_clusters_in_I_p = set(self.antecedents_indices_for_each_rule[:,p])
            self.J[p] = len(fuzzy_clusters_in_I_p)
            self.total_antecedents += self.J[p]
        
        # between inputs and antecedents
        self.W_1 = np.zeros((self.P, self.total_antecedents))
        start_idx = 0
        for p in range(self.P):
            end_idx = start_idx + self.J[p]
            self.W_1[p, start_idx:end_idx] = 1
            start_idx = end_idx
            # print(W_1[p])
        
        # between antecedents and rules
        self.W_2 = np.empty((self.total_antecedents, self.K))
        self.W_2[:] = np.nan
        for rule_index, antecedents_indices_for_rule in enumerate(self.antecedents_indices_for_each_rule):
            start_idx = 0
            for input_index, antecedent_index in enumerate(antecedents_indices_for_rule):
                self.W_2[start_idx + antecedent_index, rule_index] = 1
                start_idx += self.J[input_index]
        
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
        self.L = {}
        self.total_consequents = 0
        for q in range(self.Q):
            fuzzy_clusters_in_O_q = set(self.consequents_indices_for_each_rule[:,q])
            self.L[q] = len(fuzzy_clusters_in_O_q)
            self.total_consequents += self.L[q]
        
        # between rules and consequents
        self.W_3 = np.empty((self.K, self.total_consequents))
        self.W_3[:] = np.nan
        for rule_index, consequent_indices_for_rule in enumerate(self.consequents_indices_for_each_rule):
            start_idx = 0
            for output_index, consequent_index in enumerate(consequent_indices_for_rule):
                self.W_3[rule_index, start_idx + consequent_index] = 1
                start_idx += self.L[output_index]
                
        consequent_activations = np.swapaxes(np.multiply(o3, self.W_3.T[:, np.newaxis]), 0, 1)
        self.f4 = np.nanmax(consequent_activations, axis=2)
        return self.f4
    
    def output_layer(self, o4):
        numerator = np.nansum((o4 * self.term_dict['consequent_centers'] * self.term_dict['consequent_widths']), axis=1)
        denominator = np.nansum((o4 * self.term_dict['consequent_widths']), axis=1)
        self.f5 = numerator / denominator
        return self.f5

    def feedforward(self, x):
        self.o1 = self.input_layer(x)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2)
        self.o4 = self.consequence_layer(self.o3)
        self.o5 = self.output_layer(self.o4)
        return self.o5