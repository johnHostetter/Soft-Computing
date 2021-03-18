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
    from rule import LogisticTerm, FLC_Rule
except ImportError:
    from ann import ANN
    from apfrb import APFRB
    from rule import LogisticTerm, FLC_Rule
    
def T(ann):
    """
    Defines the transformation between ANN to APFRB.

    Returns
    -------
    APFRB
        This ANN's equivalent APFRB.

    """
    a_0 = ann.beta # assuming activation function is tanh
    a = [a_0]
    a.extend(ann.c)
    v = -1.0 * ann.b
    return APFRB(ann.W, v, a)

def T_inv(apfrb):

    """
    Defines the inverse transformation between APFRB to ANN.

    Returns
    -------
    ANN
        This APFRB's equivalent ANN.

    """
    beta = apfrb.a[0] # fetching the output node's bias
    try:
        b = -1.0 * apfrb.v # returning the biases to their original value
    except TypeError: # self.v is saved as a sequence instead of a numpy array
        b = -1.0 * np.array(apfrb.v)
    c = apfrb.a[1:] # fetching the weights between the hidden layer and the output node
    return ANN(apfrb.W, b, c, beta)

def APFRB_rule_to_FLC_rule(apfrb_rule):
    antecedents = {}
    indices = list(apfrb_rule.antecedents.keys())
    values = list(apfrb_rule.antecedents.values())
    for loop_idx in range(len(values)):
        index = indices[loop_idx]
        entry = values[loop_idx]
        k = apfrb_rule.v[index - 1]
        if entry:
            antecedents[index] = LogisticTerm(k, '+')
        else:
            antecedents[index] = LogisticTerm(k, '-')
    return FLC_Rule(antecedents, apfrb_rule.consequent())

def APFRB_to_FLC(apfrb):
    pass