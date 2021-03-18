#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:25:46 2021

@author: john
"""

import numpy as np

class FLC:
    def __init__(self, rules, table):
        self.rules = rules
        self.table = table
        self.d_memo = {}
    def __u(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        u = 0.0
        q = len(self.rules)
        for i in range(q):
            u += self.rules[i].t(x) * self.rules[i].consequent
        return u

    def __d(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        d : TYPE
            DESCRIPTION.

        """
        key = hash(tuple(map(float, x)))
        if key in self.d_memo:
            return self.d_memo[key]
        else:
            d = 0.0
            q = len(self.rules)
            for i in range(q):
                d += self.rules[i].t(x)
            self.d_memo[key] = d
            return d
    def infer_with_u_and_d(self, x):
        """
        Conducts the FLC's fuzzy inference and defuzzification when given an input 'x'.
        Capable of execution when the FLC is no longer equivalent to its previous APFRB/ANN.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.__u(x) / self.__d(x)
    def predict(self, D, func, ann):
        predictions = []
        for z in D:
            y = []
            for j in range(ann.m):
                y.append(np.dot(ann.W[j].T, z))
            x = dict(zip(range(1, len(y) + 1), y))
            f = self.infer_with_u_and_d(x)
            prediction = func(f)
            predictions.append(prediction)
        return np.array(predictions)