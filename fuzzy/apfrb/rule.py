#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:51:30 2021

@author: john
"""

import numpy as np
from sympy import Symbol, N # N is used to evaluate floating point approximations
from copy import deepcopy

from sklearn import datasets

try:
    from .flc import FLC
    from .common import subs, logistic
except ImportError:
    from flc import FLC
    from common import subs, logistic
    
class OrdinaryTerm:
    def __init__(self, sympy_expr_interval, z_i, precision=4):
        """
        z_i is the raw data index that the sympy expression interval is conditioned on

        Parameters
        ----------
        sympy_expr_interval : TYPE
            DESCRIPTION.
        z_i : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sympy_expr_interval = sympy_expr_interval
        self.z_i = z_i
        self.precision = precision
        
    def __str__(self):
        return str(N(self.sympy_expr_interval, self.precision))
    
    def mu(self, x):
        """
        where x is a raw observation

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        expr = deepcopy(self.sympy_expr_interval)
        for i, z_i in enumerate(self.z_i):
            arg = 'z[%s]' % z_i
            expr = expr.subs(Symbol(arg), x[int(z_i)])
            
        return 1.0 if expr else 0.0

class LogisticTerm:
    def __init__(self, k, neg_or_pos):
        self.k = k
        self.type = neg_or_pos
        self.__logistic = logistic
        self.memo = {}
        
    def __hash__(self):
        if self.type == '+':
            return hash(self.k)
        else:
            return hash(-self.k)
        
    def __str__(self):
        if self.type == "+":
            return ("larger than %.2f" % self.k)
        else:
            return ("smaller than %.2f" % self.k)
        
    def mu(self, x):
        return self.__logistic(x, self.k, self.type)
    
    def export(self):
        return {'id':hash(self), 'k':self.k, 'type':self.type}

class FLC_Rule:
    def __init__(self, antecedents, consequents, else_clause=None):
        self.antecedents = antecedents
        self.consequents = consequents
        self.else_clause = else_clause # by default is None, but may contain some rules when ordering matters
        self.default_class = False # by default is False, becomes True when this rule's consequent is the default class
        self.ordinary_logic = False # by default is False, becomes True when rule is either valid or not
        self.threshold = 0.9 # the activation threshold required for the rule to be valid
        self.memo = {}
        
    def __str__(self):
        if self.default_class:
            return str(self.consequent())
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
        temp = (' THEN f(x) = %.2f' % self.consequent())
        output += temp
        if self.else_clause is not None:
            output += ' ELSE;'
            output += '\n'
            output += str(self.else_clause)
        return output
    
    def consequent(self):
        if isinstance(self.consequents, float):
            return self.consequents
        else:
            raise Exception('FLC rule consequent is not a float.')
            
    def __crisp_logic(self, x, degree):
        if not self.ordinary_logic:
            return degree
        else:
            if degree < self.threshold:
                try:
                    return self.else_clause.t(x)
                except AttributeError:
                    if isinstance(self.else_clause, FLC):
                        return self.else_clause.rules[0].t(x)
                    elif isinstance(self.else_clause, float): # default class
                        return self.else_clause
                # return 0
            else:
                return self.consequent()

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
            return self.__crisp_logic(x, self.memo[key])
        else:
            degree = 1.0
            for key in self.antecedents.keys():
                if isinstance(self.antecedents[key], OrdinaryTerm):
                    degree *= self.antecedents[key].mu(list(x.values()))
                else:
                    degree *= self.antecedents[key].mu(x[key])
            self.memo[key] = degree
            return self.__crisp_logic(x, degree)
    
    def export(self):
        rule = {}
        for key in self.antecedents.keys():
            col_name = '{}'.format(key)
            rule[col_name] = hash(self.antecedents[key])
        rule['consequent'] = self.consequent()
        return rule

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
        float
            The degree of applicability of this rule's antecedents.

        """
        key = hash(tuple(map(float, z)))
        if key in self.memo:
            return self.memo[key]
        else:
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
