#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:41:23 2021

@author: john
"""

import matplotlib.pyplot as plt

from fuzzy.denfis.ecm import ECM
from sklearn.datasets import make_blobs

SUPPRESS_EXCEPTIONS = True
centers = [(-5, -5), (5, 5)]
cluster_std = [0.8, 1]

X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")

Cs = ECM(X, [], 3)
