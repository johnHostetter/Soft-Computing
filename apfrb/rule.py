#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:51:30 2021

@author: john
"""

import numpy as np
from common import subs

class LogisticTerm:
    def __init__(self, k, neg_or_pos):
        self.k = k
        self.type = neg_or_pos
        self.memo = {}
    def __str__(self):
        if self.type == "+":
            return ("larger than %s" % self.k)  
        else:
            return ("smaller than %s" % self.k)
    def mu(self, x):
        key = hash(tuple(map(float, x)))
        if key in self.memo:
            return self.memo[key]
        else:
            self.memo[key] = self.logistic(x, self.k, self.type)
            return self.memo[key]
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

class ElseRule:
    def __init__(self, consequent):
        self.consequent = consequent
    def __str__(self):
        return 'ELSE %s' % self.consequent

class FLC_Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent
        self.else_clause = False # by default is False, but may become True when rule ordering matters
    def __str__(self):
        output = 'IF'
        iterations = 0
        limit = len(self.antecedents.keys())
        for key in self.antecedents.keys():
            antecedent = self.antecedents[key]
            output += ' X_%s is ' % key
            output += str(antecedent)
            if iterations < limit - 1:
                output += ' AND'
            iterations += 1
        temp = (' THEN f(x) = %s' % self.consequent)
        output += temp
        if self.else_clause:
            output += ' ELSE '
        return output
    def t(self, x):
        """
        Calculates the degree of firing for this rule.

        Parameters
        ----------
        x : dictionary
            The FLC's input.

        Returns
        -------
        None.

        """
        pass

class Rule:
    def __init__(self, antecedents, consequents, lookup, W, v):
        self.antecedents = antecedents # dictionary
        self.consequents = consequents # dictionary
        self.lookup = lookup # lookup table for the term's linguistic meaning
        self.W = W
        self.v = v
        self.memo = {}
    
    def __str__(self):
        """
        Generates a string representation of the fuzzy logic rule.

        Returns
        -------
        output : string
            A string representation of the fuzzy logic rule.

        """
        indices = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        
        a = list(self.consequents.values())
        signs = list(map(subs, values)) # a vector describing the +/- signs for the a's in the IF-THEN consequents
        consequent = a[0] + np.dot(signs, a[1:]) # TODO: consider storing the consequent in the rule
        output = 'IF '
        for loop_idx in range(len(values)):
            index = indices[loop_idx]
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
        key = hash(tuple(map(float, z)))
        try:
            return self.memo[key]
        except KeyError:
            degree = 1.0
            indices = list(self.antecedents.keys())
            values = list(self.antecedents.values())
            for loop_idx in range(len(values)):
                index = indices[loop_idx]
                entry = values[loop_idx]
                # y = x[index - 1]
                y = np.dot(self.W[index - 1].T, z)
                k = self.v[index - 1]
                if entry:
                    degree *= self.logistic(y, k, '+')
                else:
                    degree *= self.logistic(y, k)
            self.memo[key] = degree
            return degree
    
    def convert_to_flc_type(self):
        antecedents = {}
        indices = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        for loop_idx in range(len(values)):
            index = indices[loop_idx]
            entry = values[loop_idx]
            k = self.v[index - 1]
            if entry:
                antecedents[index] = LogisticTerm(k, '+')
            else:
                antecedents[index] = LogisticTerm(k, '-')
        return FLC_Rule(antecedents, self.consequent())
                
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