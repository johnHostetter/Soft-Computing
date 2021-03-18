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
    from .ann import iris_ann
    from .transformation import T
    from .rule_reduction import RuleReducer
except ImportError:
    from ann import iris_ann
    from transformation import T
    from rule_reduction import RuleReducer
    
np.random.seed(10)

def norm_z(z, z_min, z_max):
    return (z - z_min) / (z - z_max)

def iris_classification(f):
    if f < -0.5:
        return 1 # versicolor
    elif -0.5 < f and f < 0.5:
        return 2 # virginica
    elif 0.5 < f:
        return 0 # setosa

def avg_error(apfrb, ann, D):
    errors = []
    for x in D:
        errors.append(abs(apfrb.inference(x) - ann.forward(x)))
    return np.mean(errors)

def read(equations):
    for equation in equations:
        print(sp.sympify(equation))
    
def infer(flc, Z, i):
    z = Z[i]
    y = []
    for j in range(ann.m):
        y.append(np.dot(ann.W[j].T, z))
    x = dict(zip(range(1, len(y) + 1), y))
    return flc.infer_with_u_and_d(x)

if __name__ == '__main__':
    ann = iris_ann()
    apfrb = T(ann)
    print(apfrb.r)

    # import some data to play with
    iris = datasets.load_iris()
    Z = iris.data[:, :4]  # we only take the first four features.
    Z = np.flip(Z, axis = 1)
    # y = iris.target - 1 # target values that match APFRB paper
    labels = iris.target
    
    start = time.time()
    ruleReducer = RuleReducer(apfrb)
    flc = ruleReducer.to_flc(Z, True)
    # rules, intervals, equations, reduced = ruleReducer.simplify(Z, MULTIPROCESSING=False)
    z = Z[0]
    y = []
    for j in range(ann.m):
        y.append(np.dot(ann.W[j].T, z))
    x = dict(zip(range(1, len(y) + 1), y))
    mu = flc.rules[0].t(x)
    end = time.time()
    print('%s seconds' % (end-start))
