#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:04:00 2021

@author: john
"""

import itertools
import numpy as np
from copy import deepcopy

try:
    from .flc import FLC
    from .rule import APFRB_Rule
    from .common import logistic
except ImportError:
    from flc import FLC
    from rule import APFRB_Rule
    from common import logistic

class APFRB(FLC):
    def __init__(self, W, v, a):
        """
        Create an All-Permutations Fuzzy Rule Base (APFRB).

        Parameters
        ----------
        W : 2-dimensional Numpy array
            The weights between the raw inputs of the ANN and the ANN's hidden layer.
        v : 1-dimensional Numpy array
            The biases for the ANN's hidden layer.
        a : 1-dimensional Numpy array
            The weights between the ANN's hidden layer and output node.
            WARNING: first entry is the bias for the output node.

        Returns
        -------
        None.

        """
        self.W = W # the weights between the raw inputs and the hidden layer
        self.v = list(v) # a vector of size m describing the biases for the ANN's hidden layer
        self.a = a # a vector of size m + 1 (since it includes a_0 - the output node's bias)
        self.__reset() # reset/initialize all the variables that are dependent upon 'W', 'v' or 'a'

    def __deepcopy__(self, memo):
        """
        Returns a deep copy of the original APFRB.

        Parameters
        ----------
        memo : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        W = deepcopy(self.W)
        v = deepcopy(self.v)
        a = deepcopy(self.a)
        return APFRB(W, v, a)

    def __reset(self):
        """
        Resets all of the APFRB's variables excluding the matrix 'W',
        the vector 'v' and the vector 'a'. These variables are to be reset
        if 'W', 'v' or 'a' are modified at any point.

        Returns
        -------
        None.

        """
        self.linguistic_terms = 'log' # TODO: add option to make Gaussian membership functions
        self.l = len(self.W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
        self.r = pow(2, self.l) # the number of fuzzy logic rules for all permutations
        self.m = len(self.v) # the number of neurons in the hidden layer
        self.n = len(self.W[0]) # the number of raw inputs
        self.lookup = {}
        self.logistic_terms = ['smaller than', 'larger than']
        
        # create the table, repeating the number of times there are rule antecedents
        FLC.__init__(self, [], list(itertools.product([False, True], repeat=self.l)))
                
        if self.n > self.m:
            size = self.n
        else:
            size = self.m
        
        for key in range(size):
            self.lookup[key] = self.logistic_terms
        
        for i in range(self.r):
            rule = self.table[i] # only contains the antecedents' term assignments
            antecedents = {(key + 1): value for key, value in enumerate(rule)} # indexed by x_i
            consequents = {key: value for key, value in enumerate(self.a)} # indexed by a_i, including a_0
            self.rules.append(APFRB_Rule(antecedents, consequents, self.lookup, self.W, self.v))

    def __delete(self, i):
        """
        Deletes the i'th entry from vector 'v' and matrix 'W', and deletes
        the i'th + 1 entry from vector 'a' (to skip the a_0 entry aka the output's bias).

        CAUTION: This mutates the APFRB calling the private method. Use with extreme caution.

        Parameters
        ----------
        i : int
            i'th entry of vector 'v', matrix 'W', and i'th + 1 entry of vector 'a'.

        Returns
        -------
        None.

        """
        self.W = np.delete(self.W, i, axis=0)
        self.v = np.delete(self.v, i, axis=0)
        self.a = list(np.delete(self.a, i + 1, axis = 0))
        self.__reset()

    def c_k(self, x, k):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        diffs = []
        rule_k = self.rules[k]
        for i in range(len(self.rules)):
            rule_i = self.rules[i]
            diffs.append(abs(rule_i.consequent() - rule_k.consequent()))
        return (1/self.d(x)) * max(diffs)

    def __b(self, x, k):
        diffs = []
        f_k = self.rules[k].t(x)
        for i in range(len(self.rules)):
            f_i = self.rules[i].t(x)
            diffs.append(abs(f_i - f_k))
        return max(diffs)

    def predict(self, D, func):
        predictions = []
        for z in D:
            f = self.infer_with_u_and_d(z)
            prediction = func(f)
            predictions.append(prediction)
        return np.array(predictions)

    def infer_with_ann(self, z):
        """
        Conducts the APFRB's fuzzy inference and defuzzification when given a raw input 'z'.

        CAUTION: This may no longer work after simplifying the APFRB.

        Parameters
        ----------
        z : list
            Raw input provided to the ANN/APFRB.

        Raises
        ------
        Exception
            An exception is thrown when the error tolerance exceeds a constant value.

        Returns
        -------
        f : float
            Crisp output.

        """
        epsilon = 1e-6 # error tolerance between flc output and ann output
        f = self.a[0]
        x = []
        for j in range(self.m):
            y = np.dot(self.W[j].T, z)
            x.append(y)
            t = np.tanh(x[j] - self.v[j]) # ann formula
            if True: # disable if not interested in checking FLC consistency
                # check FLC inference is still consistent with ann formula
                k = self.v[j]
                t_num = logistic(y, k, '+') - logistic(y, k)
                t_den = logistic(y, k, '+') + logistic(y, k)
                t_flc = t_num / t_den
                if abs(t_flc - t) >= epsilon:
                    raise Exception('The error tolerance of epsilon has been violated in the APFRB\'s inference.')
            j += 1 # look-ahead by 1 (to avoid the first entry which is the output node's bias)
            f += self.a[j] * t
        return f