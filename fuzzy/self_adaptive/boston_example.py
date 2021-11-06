#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:28:08 2021

@author: john
"""

import numpy as np

from sklearn.datasets import load_boston

from fuzzy.self_adaptive.safin import SaFIN

def main():
    boston = load_boston()
    NUM_DATA = 400
    train_X = boston.data[:NUM_DATA]
    train_Y = np.array([boston.target]).T[:NUM_DATA]
    test_X = boston.data[NUM_DATA:]
    test_Y = np.array([boston.target]).T[NUM_DATA:]
    safin = SaFIN(alpha=0.2, beta=0.6)
    _ = safin.fit(train_X, train_Y, batch_size=50, epochs=10, verbose=False, rule_pruning=False)
    rmse = safin.evaluate(test_X, test_Y)
    print('Test RMSE = %.6f' % rmse)
    return safin

if __name__ == '__main__':
    safin = main()
