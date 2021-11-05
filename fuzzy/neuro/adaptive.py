#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:19:15 2021

@author: john
"""

import time
import numpy as np

from fuzzy.neuro.core import CoreNeuroFuzzy
from fuzzy.common.utilities import boolean_indexing
    
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
        CoreNeuroFuzzy.__init__(self)
        self.rules = [] # the fuzzy logic rules
        self.weights = [] # the weights corresponding to the rules, the i'th weight is associated with the i'th rule
        self.antecedents = []
        self.consequents = []
        self.P = None
        self.Q = None
        self.K = 0
        
    def get_number_of_rules(self):
        return self.K
    
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
        antecedents : 2-D list
            The parameters for the antecedent terms. 
            The first index applied (e.g. antecedents[i] where 0 <= i <= number of inputs/attributes) 
            will return the antecedent terms for the i'th input/attribute.
            The second index applied (e.g. antecedents[i][j] where 0 <= i <= number of inputs/attributes 
                                      and 0 <= j <= number of antecedent terms for the i'th input/attribute)
                                      will return the j'th antecedent term for the i'th input/attribute.
        consequents : 2-D list
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
