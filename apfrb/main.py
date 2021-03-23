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
    from .ann import iris_ann, ANN
    from .transformation import T
    from .rule_reduction import RuleReducer
except ImportError:
    from ann import iris_ann, ANN
    from transformation import T
    from rule_reduction import RuleReducer
    
np.random.seed(10)

def norm_z(z, z_min, z_max):
    return (z - z_min) / (z - z_max)

def iris_labels(f):
    """
    The target labels for the sklearn's Iris data set do not match the 
    APFRB paper's labels. Therefore, this function defines the mapping to 
    transform the sklearn's labels to match those of the original paper.
    
    In sklearn, Setosa is represented by 0, Versicolor by 1, and Virginica by 2.

    Parameters
    ----------
    f : float
        Iris target label.

    Returns
    -------
    float
        A float encoding of the class label for the Iris classification 
        problem identical to what was used in the APFRB paper.

    """
    return 1 if f == 0 else -1 if f == 1 else 0

def iris_classification(f):
    """
    Versicolor is represented by -1.
    Virginica is represented by 0.
    Setosa is represented by 1.
    
    Note: Behaves identically to numpy.round(), but restricts the result to [-1, 1].

    Parameters
    ----------
    f : float
        Crisp output from prediction or inference.

    Returns
    -------
    float
        A float encoding of the class label for the Iris classification 
        problem identical to what was used in the APFRB paper.

    """
    return -1 if f < -0.5 else 1 if 0.5 < f else 0

def avg_error(apfrb, ann, D):
    errors = []
    for x in D:
        errors.append(abs(apfrb.inference(x) - ann.forward(x)))
    return np.mean(errors)

def read(equations):
    for equation in equations:
        print(sp.sympify(equation))
    
def infer(flc, ann, Z, i):
    z = Z[i]
    y = []
    for j in range(ann.m):
        y.append(np.dot(ann.W[j].T, z))
    x = dict(zip(range(1, len(y) + 1), y))
    return flc.infer_with_u_and_d(x)

def pyrenees_ann():
    from tensorflow.keras.models import load_model
    model = load_model('test_model_classification.h5')
    return ANN(model.trainable_weights[0].numpy().T, model.trainable_weights[1].numpy().T, 
              model.trainable_weights[2].numpy().T[0], model.trainable_weights[3].numpy()[0])

def iris_example():
    ann = iris_ann()
    apfrb = T(ann)
    print(apfrb.r)

    # import some data to play with
    iris = datasets.load_iris()
    Z = iris.data[:, :4]  # we only take the first four features.
    Z = np.flip(Z, axis = 1)
    # from sklearn.preprocessing import Normalizer
    # scaler = Normalizer().fit(Z)
    # Z = scaler.transform(Z)
    labels = np.array([iris_labels(label) for label in iris.target]) # target values that match APFRB paper
    
    # apfrb.predict_with_ann(Z, ann, iris_classification)
    
    # np.round([apfrb.inference(z) for z in Z])
    
    # start = time.time()
    ruleReducer = RuleReducer(apfrb)
    flc = ruleReducer.to_flc(Z, True)
        
    # rules, intervals, equations, reduced = ruleReducer.simplify(Z, MULTIPROCESSING=False)
    
    from common import foo, foobar
    
    ordered_table, ordered_rules = foo(flc.table, flc.rules)
    
    filtered_rules = foobar(ordered_table, ordered_rules)
    rules = filtered_rules
    
    from common import barfoo
    
    result = barfoo(ordered_table, rules)
    
    # z = Z[0]
    # y = []
    # for j in range(ann.m):
    #     y.append(np.dot(ann.W[j].T, z))
    # x = dict(zip(range(1, len(y) + 1), y))
    # mu = flc.rules[0].t(x)
    # end = time.time()
    # print('%s seconds' % (end-start))
    # print(flc)
    return apfrb, ann, rules, ordered_table, ruleReducer, result
    
def pyrenees():
    ann = pyrenees_ann()
    apfrb = T(ann)
    print(apfrb)
    return apfrb

if __name__ == '__main__':
    apfrb, ann, rules, tables, reducer, result = iris_example()
    from common import barbar
    res = barbar(result)
    # import some data to play with
    iris = datasets.load_iris()
    labels = np.array([iris_labels(label) for label in iris.target]) # target values that match APFRB paper
    Z = iris.data[:, :4]  # we only take the first four features.
    Z = np.flip(Z, axis = 1)
    hflc_pred = []
    for z in Z:
        y = []
        for j in range(ann.m):
            y.append(np.dot(ann.W[j].T, z))
        x = dict(zip(range(1, len(y) + 1), y))
        hflc_pred.append(res.t(x))
    