#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:51:30 2021

@author: john
"""

import numpy as np
from common import subs

class Rule:
    def __init__(self, antecedents, consequents, lookup, W, v):
        self.antecedents = antecedents # dictionary
        self.consequents = consequents # dictionary
        self.lookup = lookup # lookup table for the term's linguistic meaning
        self.W = W
        self.v = v
    
    def __str__(self):
        """
        Generates a string representation of the fuzzy logic rule.

        Returns
        -------
        output : string
            A string representation of the fuzzy logic rule.

        """
        indexes = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        
        a = list(self.consequents.values())
        signs = list(map(subs, values)) # a vector describing the +/- signs for the a's in the IF-THEN consequents
        consequent = a[0] + np.dot(signs, a[1:]) # TODO: consider storing the consequent in the rule
        output = 'IF '
        for loop_idx in range(len(values)):
            index = indexes[loop_idx]
            entry = values[loop_idx]
            if entry: # term+ is present
                output += ('x_%s is %s %.2f ' % (index, self.lookup[index - 1][1], self.v[index - 1]))
            else:
                output += ('x_%s is %s %.2f ' % (index, self.lookup[index - 1][0], self.v[index - 1]))
            if loop_idx != len(values) - 1:
                output += 'AND '
        output += 'THEN f = %.2f' % (consequent) # the consequent for the IF-THEN rule
        return output
    
    def normalize_keys(self):
        """
        Resets the keys in the rule to begin count from '1'.

        Returns
        -------
        None.

        """
        rule = list(self.antecedents.values())
        a = list(self.consequents.values())
        self.antecedents = {(key + 1): value for key, value in enumerate(rule)}
        self.consequents = {key: value for key, value in enumerate(a)}
    
    def consequent(self):
        values = list(self.antecedents.values())
        a = list(self.consequents.values())
        signs = list(map(subs, values)) # a vector describing the +/- signs for the a's in the IF-THEN consequents
        return a[0] + np.dot(signs, a[1:])
    
    def t(self, z):
        """
        Calculates the degree of firing for this rule.

        Parameters
        ----------
        z : list
            The ANN's input.

        Returns
        -------
        None.

        """
        degree = 1.0
        indexes = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        for loop_idx in range(len(values)):
            index = indexes[loop_idx]
            entry = values[loop_idx]
            # y = x[index - 1]
            y = np.dot(self.W[index - 1].T, z)
            k = self.v[index - 1]
            if entry:
                degree *= self.logistic(y, k, '+')
            else:
                degree *= self.logistic(y, k)
        return degree
                
    def logistic(self, y, k, t='-'):
        """
        The logistic membership function.

        Parameters
        ----------
        y : float
            The input.
        k : float
            The bias (for logistic functions, k_i = v_i for all i in 1, 2, ... m).
        t : string, optional
            Adjusts whether this logistic membership function is describing term- or term+. 
            The default is '-'.

        Returns
        -------
        float
            The degree of membership.

        """
        val = 2.0
        if t == '+':
            val = -2.0
        return 1.0 / (1.0 + np.exp(val * (y-k)))