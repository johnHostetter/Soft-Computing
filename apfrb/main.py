#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:49:48 2021

@author: john
"""

import time
import numpy as np
import sympy as sp
from sklearn import datasets

try:
    from .ann import ANN
    from .rule_reduction import RuleReducer
    from .transformation import T
except ImportError:
    from ann import ANN
    from rule_reduction import RuleReducer
    from transformation import T
    
np.random.seed(10)

def norm_z(z, z_min, z_max):
    return (z - z_min) / (z - z_max)

def main():
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
    # n_inputs = 4 # number of inputs, has no impact on program executability
    # n_neurons = 10 # number of neurons in the hidden layer, maximum number of neurons this can handle is 21
    # W = np.random.random(size=(n_neurons, n_inputs))
    # b = np.random.random(size=(n_neurons,))
    # c = np.random.random(size=(n_neurons,))
    if len(b) != len(c):
        raise Exception('The vector \'b\' must equal the vector \'c\'.')
    l = len(W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
    r = pow(2, l) # the number of fuzzy logic rules for all permutations
    return ANN(W, b, c, 0.0), l, r

def iris_classification(f):
    if f < -0.5:
        return -1 # versicolor
    elif -0.5 < f and f < 0.5:
        return 0 # virginica
    elif 0.5 < f:
        return 1 # setosa

def avg_error(apfrb, ann, D):
    errors = []
    for x in D:
        errors.append(abs(apfrb.inference(x) - ann.forward(x)))
    return np.mean(errors)

def read(equations):
    for equation in equations:
        print(sp.sympify(equation))

if __name__ == '__main__':
    ann, l, r = main()
    apfrb = T(ann)
    print(apfrb.r)

    # import some data to play with
    iris = datasets.load_iris()
    Z = iris.data[:, :4]  # we only take the first four features.
    Z = np.flip(Z, axis = 1)
    y = iris.target - 1 # target values that match APFRB paper

    start = time.time()
    ruleReducer = RuleReducer(apfrb)
    ruleReducer.simplify(Z)
    end = time.time()
    print(end-start)
