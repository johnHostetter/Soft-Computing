#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:49:11 2021

@author: john
"""

import numpy as np

from scipy.spatial.distance import minkowski

def general_euclidean_distance(x, y):
    if len(x) == len(y): 
        q = len(x)
        return minkowski(x, y, p=2) / np.power(q, 0.5)
    else:
        raise TypeError('The vectors must of of equal dimensionality in order to use the General Euclidean Distance metric.')