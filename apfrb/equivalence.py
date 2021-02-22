#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:22:10 2021

@author: john
"""

import numpy as np

def logistic_n(y, k):
    return 1 / (1 + np.exp(2 * (y-k)))
        
def logistic_p(y, k):
    return 1 / (1 + np.exp(-2 * (y-k)))

def gaussian(y, k):
    return np.exp(-1.0 * (pow(y - k, 2)) / (2 * k))

def gaussian_n(y, k):
    return np.exp(-1.0 * (pow(y - k[1], 2)) / (k[0] - k[1]))

def gaussian_p(y, k):
    return np.exp(-1.0 * (pow(y - k[0], 2)) / (k[0] - k[1]))

# --- example 1 ---

# initialization
a_0 = 1
a_1 = 2.4
x = 1.0
k = 3.5

# formula for the ANN
g = a_0 + a_1 * np.tanh(x)

# formula for APFRB
# TODO: fix this equivalence
f = (a_0 + a_1)*gaussian(x, k) + (a_0 - a_1)*gaussian(x, -k)
f /= gaussian(x, k) + gaussian(x, -k)

print('The difference between example 1\'s formula is = %s' % (f - g))

# --- example 2 ---

# initialization
x = [1.0, 1.0]
a_0 = 1.0
a_1 = 1/3
a_2 = 2/5
x1_k = 5
x2_k = [7, 1]

# formula for ANN
g = a_0 + np.tanh(x[0] - 5)/3 + (2*(np.tanh(x[1] - 4)/5))

# formula for APFRB
input1 = (logistic_p(x[0], 5) - logistic_n(x[0], 5)) / (logistic_p(x[0], 5) + logistic_n(x[0], 5))
input2 = (gaussian_p(x[1], x2_k) - gaussian_n(x[1], x2_k)) / (gaussian_p(x[1], x2_k) + gaussian_n(x[1], x2_k))
f = a_0 + a_1 * input1 + a_2 * input2

print('The difference between example 2\'s formula is = %s' % (f - g))