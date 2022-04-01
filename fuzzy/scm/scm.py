#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:53:44 2021

@author: john
"""

import numpy as np

from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances


def potential(X, r_a):
    P = np.zeros(X.shape[0]) # the potential of each data point
    for i, x_i in enumerate(X):
        dist = np.power(euclidean_distances(x_i[np.newaxis, :], X), 2)
        alpha = 4 / np.power(r_a, 2)
        P[i] = (np.power(np.e, -1 * alpha * dist)).sum()
    return P


def SCM(X, Y, r_a=0.1):
    clusters = []
    
    # first cluster
    P = potential(X, r_a)
    
    first_cluster_potential = np.max(P)
    first_cluster_idx = np.argmax(P)
    first_cluster_center = X[first_cluster_idx]
    
    clusters.append(first_cluster_center)
    max_potential = first_cluster_potential # starting condition
    last_cluster_center = first_cluster_center
    
    while max_potential > 0.15 * first_cluster_potential:   
        r_b = 1.5 * r_a
        kth_P = deepcopy(P)
        # repeat the above
        P = potential(X, r_a)
        
        beta = 4 / np.power(r_b, 2)
        dist = np.power(euclidean_distances(X, last_cluster_center[np.newaxis, :]), 2)
        P = P - kth_P * (np.power(np.e, -1 * beta * dist)).sum()
        
        max_potential = np.max(P)
        second_cluster_idx = np.argmax(P)
        second_cluster_center = X[second_cluster_idx]
        clusters.append(second_cluster_center)
        last_cluster_center = second_cluster_center

    return clusters, P
