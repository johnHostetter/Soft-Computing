#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:48:59 2021

@author: john
"""

import numpy as np

try:
    from .main import APFRB
except ImportError:
    from main import APFRB

class ANN:
    def __init__(self, W, b, c, beta):
        """
        Create an Artificial Neural Network (ANN).

        Parameters
        ----------
        W : 2-dimensional Numpy array
            The weights between the raw inputs of the ANN and the ANN's hidden layer.
        b : 1-dimensional Numpy array
            The biases for the ANN's hidden layer.
        c : 1-dimensional Numpy array
            The weights between the ANN's hidden layer and output node.
        beta : float
            The bias influencing the ANN's output node.

        Returns
        -------
        None.

        """
        self.W = W # the weights between the raw inputs and the hidden layer
        self.b = b # the biases for the hidden layer
        self.c = c # the weights between the hidden layer and the output node
        self.beta = beta # the bias for the output node
        self.m = len(self.b) # the number of neurons in the hidden layer
        self.n = len(self.W[0]) # the number of raw inputs

    def forward(self, z):
        """
        Conduct a forward pass in the ANN.

        Parameters
        ----------
        z : list
            Raw input provided to the ANN/APFRB.

        Returns
        -------
        f : float
            Crisp output calculated by the ANN.

        """
        f = self.beta
        y = []
        for j in range(self.m):
            y.append(np.dot(self.W[j].T, z))
            f += self.c[j] * np.tanh(y[j] + self.b[j])
        return f

    def T(self):
        """
        Defines the transformation between ANN to APFRB.

        Returns
        -------
        APFRB
            This ANN's equivalent APFRB.

        """
        a_0 = self.beta # assuming activation function is tanh
        a = [a_0]
        a.extend(self.c)
        v = -1.0 * self.b
        return APFRB(self.W, v, a)