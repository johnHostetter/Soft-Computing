#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:06:56 2021

@author: john
"""

import numpy as np

try:
    from .ann import ANN
    from .apfrb import APFRB
except ImportError:
    from ann import ANN
    from apfrb import APFRB
    
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

def T_inv(self):

    """
    Defines the inverse transformation between APFRB to ANN.

    Returns
    -------
    ANN
        This APFRB's equivalent ANN.

    """
    beta = self.a[0] # fetching the output node's bias
    try:
        b = -1.0 * self.v # returning the biases to their original value
    except TypeError: # self.v is saved as a sequence instead of a numpy array
        b = -1.0 * np.array(self.v)
    c = self.a[1:] # fetching the weights between the hidden layer and the output node
    return ANN(self.W, b, c, beta)