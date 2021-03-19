#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:51:30 2021

@author: john
"""

import numpy as np

try:
    from .common import subs, logistic
except ImportError:
    from common import subs, logistic

class LogisticTerm:
    def __init__(self, k, neg_or_pos):
        self.k = k
        self.type = neg_or_pos
        self.__logistic = logistic
        self.memo = {}
    def __str__(self):
        if self.type == "+":
            return ("larger than %s" % self.k)
        else:
            return ("smaller than %s" % self.k)
    def mu(self, x):
        return self.__logistic(x, self.k, self.type)

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
        self.memo = {}
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
        key = hash(tuple(map(float, list(x.values()))))
        if key in self.memo:
            return self.memo[key]
        else:
            degree = 1.0
            for key in self.antecedents.keys():
                degree *= self.antecedents[key].mu(x[key])
            self.memo[key] = degree
            return degree

class APFRB_Rule:
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
                    degree *= self.rule_mu(y, k, '+')
                else:
                    degree *= self.rule_mu(y, k)
            self.memo[key] = degree
            return degree

    def rule_mu(self, y, k, t='-'):
        # TODO: generalize this so that any linguistic term's membership function works
        return logistic(y, k, t)