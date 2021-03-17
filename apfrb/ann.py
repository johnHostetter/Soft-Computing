#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:48:59 2021

@author: john
"""

import numpy as np

def random_ann(n_inputs=3, n_neurons=6):
    """
    Prepares a random ANN that has been trained to recognize some arbitrary function.
    
    CAUTION: n_neurons cannot exceed the value of 21, whereas n_inputs does not impact
    program executability.

    Parameters
    ----------
    n_inputs : TYPE, optional
        DESCRIPTION. The default is 3.
    n_neurons : TYPE, optional
        DESCRIPTION. The default is 6.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    W = np.random.random(size=(n_neurons, n_inputs))
    b = np.random.random(size=(n_neurons,))
    c = np.random.random(size=(n_neurons,))
    if len(b) != len(c):
        raise Exception('The vector \'b\' must equal the vector \'c\'.')
    return ANN(W, b, c, 0.0)

def iris_ann():
    """
    Prepares the ANN that has been trained to recognize the Iris dataset.

    Raises
    ------
    Exception
        An exception is thrown when the vector 'b' is not equal to the vector 'c'.

    Returns
    -------
    ANN
        The ANN of interest.
    l : int
        The number of antecedents in the fuzzy logic rules.
    r : int
        The number of fuzzy logic rules for all permutations.

    """
    W = np.array([[-0.4, -5, -0.3, 0.7], [150, 150, -67, -44], [-5, 9, -7, 2]])
    b = np.array([-7, -520, -11])
    c = np.array([-0.5, 0.5, -1])
    if len(b) != len(c):
        raise Exception('The vector \'b\' must equal the vector \'c\'.')
    return ANN(W, b, c, 0.0)

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