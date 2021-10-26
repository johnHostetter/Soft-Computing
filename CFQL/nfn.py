#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:19:15 2021

@author: john
"""

import time
import numpy as np

from common import boolean_indexing, RMSE

class CoreNeuroFuzzy:
    """
        Contains core neuro-fuzzy network functionality.
        
        Handles fuzzification, antecedent matching, 
        fuzzy inference, aggregation and defuzzification.
        
        Note: Currently only supports limited options 
        (e.g. Singleton Fuzzifier, Gaussian membership functions, etc.).
        
        WARNING: The order in which these functions are called matters.
        They should not be called out of order, in fact, only the 'predict()'
        function call should be used.
    """
    def input_layer(self, x):
        """
        Singleton Fuzzifier (directly pass on the input vector to the next layer).

        Parameters
        ----------
        x : Numpy 2D array
            The input vector, has a shape of (number of observations, number of inputs/attributes).

        Returns
        -------
        Numpy 2D array
            The input vector, has a shape of (number of observations, number of inputs/attributes).

        """
        # where x is the input vector and x[i] or x_i would be the i'th element of that input vector
        # restructure the input vector into a matrix to make the condtion layer's calculations easier
        self.f1 = x
        return self.f1
    
    def condition_layer(self, o1):        
        """
        Antecedent Matching (with Gaussian membership functions).

        Parameters
        ----------
        o1 : Numpy 2D array
            The input from the first layer, most likely the input vector(s) 
            unchanged if using Singleton Fuzzifier.

        Returns
        -------
        Numpy 2D array
            The activations of each antecedent term in the second layer, 
            has a shape of (number of observations, number of all antecedents).

        """
        activations = np.dot(o1, self.W_1) # the shape is (num of inputs, num of all antecedents)
        
        flat_centers = self.term_dict['antecedent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['antecedent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values

        denominator = np.power(flat_widths, 2)
        denominator = np.where(denominator == 0.0, np.finfo(np.float).eps, denominator) # if there is a zero in the denominator, replace it with the smallest possible float value, otherwise, keep the other values
        self.f2 = np.exp(-1.0 * (np.power(activations - flat_centers, 2) / denominator))
            
        return self.f2 # shape is (num of inputs, num of all antecedents)
    
    def rule_base_layer(self, o2, inference='minimum'):     
        """
        Fuzzy Logic Rule Matching (with Minimum inference).

        Parameters
        ----------
        o2 : Numpy 2D array
            The input from the second layer, most likely the activations 
            of each antecedent term in the second layer.

        Returns
        -------
        Numpy 2D array
            The degree of applicability of each fuzzy logic rule in the third layer,
            has a shape of (number of observations, number of rules).

        """
        rule_activations = np.swapaxes(np.multiply(o2, self.W_2.T[:, np.newaxis]), 0, 1) # the shape is (num of observations, num of rules, num of antecedents)
        if inference == 'minimum':
            self.f3 = np.nanmin(rule_activations, axis=2) # the shape is (num of observations, num of rules)
        elif inference == 'product':
            self.f3 = np.nanprod(rule_activations, axis=2)
        return self.f3
    
    def consequence_layer(self, o3):   
        """
        Consequent Derivation (with Maximum T-conorm).

        Parameters
        ----------
        o3 : Numpy 2D array
            The input from the third layer, most likely the degree of applicability
            of each fuzzy logic rule in the third layer.

        Returns
        -------
        Numpy 2D array
            The activations of each consequent term in the fourth layer, 
            has a shape of (number of observations, number of consequent terms).

        """             
        consequent_activations = np.swapaxes(np.multiply(o3, self.W_3.T[:, np.newaxis]), 0, 1)
        self.f4 = np.nanmax(consequent_activations, axis=2)
        return self.f4
    
    def output_layer(self, o4):
        """
        Defuzzification (using Center of Averaging Defuzzifier).

        Parameters
        ----------
        o4 : Numpy 2D array
            The input from the fourth layer, most likely the activations of each consequent
            term in the fourth layer.

        Returns
        -------
        Numpy 2D array
            The crisp output for each output node in the fifth layer,
            has a shape of (number of observations, number of outputs).

        """
        temp_transformation = np.swapaxes(np.multiply(o4, self.W_4.T[:, np.newaxis]), 0, 1)
        
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        
        numerator = np.nansum((temp_transformation * flat_centers * flat_widths), axis=2)
        denominator = np.nansum((temp_transformation * flat_widths), axis=2)
        self.f5 = numerator / denominator
        if np.isnan(self.f5).any():
            raise Exception()
            self.f5[np.isnan(self.f5)] = 0.0 # nan values may appear if no rule in the rule base is applicable to an observation, zero out the nan values
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
        return RMSE(est_Y, Y)
    
    def backpropagation(self, x, y):
        # (1) calculating the error signal in the output layer
        
        e5_m = y - self.o5 # y actual minus y predicted
        # e5 = np.dot(e5_m, self.W_4.T) # assign the error to its corresponding output node, shape is (num of observations, num of output nodes)
        e5 = np.multiply(e5_m[:,:,np.newaxis], self.W_4.T) # shape is (num of observations, num of output nodes, num of output terms)
        error = (self.o4 * e5.sum(axis=1))
        
        # delta centers
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        # y4_k = (centers * self.W_4.T)
        # numerator = (widths * y4_k)
        widths = np.multiply(flat_widths[:,np.newaxis], self.W_4).T
        num = np.multiply(widths[np.newaxis,:,:], self.o4[:, np.newaxis,:])
        den = np.power(num.sum(axis=2), 2)
        consequent_delta_c = e5.sum(axis=1) * (num / den[:, :, np.newaxis]).sum(axis=1)
        
        # delta widths
        # c_lk = (flat_centers * self.W_4.T)
        # lhs_term = np.dot(den, c_lk)
        # rhs_term = np.multiply(num, c_lk)
        # compatible_rhs_term = rhs_term.sum(axis=1)
        # difference = lhs_term - compatible_rhs_term
        # numerator = np.multiply(self.o4, difference)
        # denominator = np.power(den, 2)
        # compatible_numerator = np.multiply(numerator[:,np.newaxis], self.W_4.T)
        # division = (compatible_numerator / denominator[:, :, np.newaxis])
        # consequent_delta_widths = division.sum(axis=1)
        
        # between consequents and outputs
        tmp = np.zeros((x.shape[0], self.Q, self.total_consequents)) # should be the same shape as self.W_4.T, but it is (num of observations, num of output nodes, num of output terms)
        # start_idx = 0
        # for q in range(self.Q):
        #     end_idx = start_idx + self.L[q]
        #     W_4[start_idx:end_idx, q] = 1
        #     start_idx = end_idx
        
        y_lk = np.swapaxes(self.o4[:,:,np.newaxis] * self.W_4, 1, 2) # shape is (num of observations, num of output nodes, num of output terms)
        c_lk = np.multiply(flat_centers[:,np.newaxis], self.W_4).T
        lhs_term = (y_lk * widths[np.newaxis,:,:])
        rhs_term = (y_lk * widths[np.newaxis,:,:] * c_lk[np.newaxis,:,:])
        for q in range(self.Q): # iterate over the output nodes
            for k in range(self.total_consequents): # iterate over their terms
                if self.W_4.T[q, k] == 1:
                    val = ((y_lk[:, q, k])[:,np.newaxis] * ((c_lk[q, k] * lhs_term.sum(axis=2)) - rhs_term.sum(axis=2)))
                    val /= np.power(lhs_term.sum(axis=2), 2)
                    tmp[:, q, k] = val[:, q]
                    
        consequent_delta_widths = e5.sum(axis=1) * tmp.sum(axis=1)
        
        return consequent_delta_c, consequent_delta_widths
    
class AdaptiveNeuroFuzzy(CoreNeuroFuzzy):
    """
        Contains functions necessary for neuro-fuzzy network creation and adaptability.
        
        Since creation and adaptation are related, they are combined into a single class.
        
        Potential for further optimization here, updates are typically carried out by
        completely recreating the neuro-fuzzy network. This is done for simplicity.
        
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
        
        Note: Currently only supports limited options 
        (e.g. second layer generation, third layer generation, fourth layer generation, etc.).
        
        WARNING: The order in which these functions are called matters.
        Use extreme care when using these functions, as some rely on the neuro-fuzzy network 
        having already processed some kind of input (e.g. they may make a reference to self.f3 or similar).
    """
    def __init__(self):
        super().__init__()
        self.rules = [] # the fuzzy logic rules
        self.weights = [] # the weights corresponding to the rules, the i'th weight is associated with the i'th rule
        self.antecedents = []
        self.consequents = []
        self.P = None
        self.Q = None
        self.K = 0
    
    def import_existing(self, rules, weights, antecedents, consequents):
        """
        Import an existing Fuzzy Rule Base.
        
        Required to be called manually after creating the AdaptiveNeuroFuzzy object
        if there are rules, antecedents, consequents, etc. that need to be used.
        
        WARNING: Bypassing this function by modifying the rules, antecedents, 
        consequents, etc. directly will, at the very least, result in incorrect fuzzy inference. 
        At worse, it should cause the program to throw an exception.

        Parameters
        ----------
        rules : list
            Each element is a dictionary representing the fuzzy logic rule (to be updated to a Rule class).
        weights : list
            Each element corresponds to a rule's weight 
            (i.e. the i'th weight belongs to the i'th fuzzy logic rule found in the rules list).
        antecedents : 2D list
            The parameters for the antecedent terms. 
            The first index applied (e.g. antecedents[i] where 0 <= i <= number of inputs/attributes) 
            will return the antecedent terms for the i'th input/attribute.
            The second index applied (e.g. antecedents[i][j] where 0 <= i <= number of inputs/attributes 
                                      and 0 <= j <= number of antecedent terms for the i'th input/attribute)
                                      will return the j'th antecedent term for the i'th input/attribute.
        consequents : 2D list
            The parameters for the consequents terms. 
            The first index applied (e.g. consequents[i] where 0 <= i <= number of outputs) 
            will return the consequents terms for the i'th output.
            The second index applied (e.g. consequents[i][j] where 0 <= i <= number of outputs 
                                      and 0 <= j <= number of consequent terms for the i'th output)
                                      will return the j'th consequent term for the i'th output.
        Returns
        -------
        None.

        """
        self.rules = rules
        self.weights = weights
        self.antecedents = antecedents
        self.consequents = consequents
        
        K = len(rules)
        if K > 0:
            self.K = K
            self.P = len(rules[0]['A'])
            self.Q = len(rules[0]['C'])
            
    def orphaned_term_removal(self):
        # need to check that no antecedent/consequent terms are "orphaned"
        # this makes sure that each antecedent/consequent term belongs to at least one fuzzy logic rule
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
    
    def preprocessing(self, verbose=False):
        # make (or update) the neuro-fuzzy network
        # note: this doesn't actually "make" the neuro-fuzzy network however,
        # it preprocesses the antecedents and consequents to be compatible with the 
        # mathematical calculations that will be used in the CoreNeuroFuzzy class functions
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
            
    def update(self):
        # this function call actually makes/updates the connections in the neuro-fuzzy network
        # preprocessing must first be called before calling update
        # the order of calls should go as follows:
        # __init__() --> load_existing() --> orphaned_term_removal() --> preprocessing() --> update()
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