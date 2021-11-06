#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:41:25 2021

@author: john
"""

import numpy as np

def weighted_RMSE(predicted_Y, target_Y):
    # return KL(predicted_Y, target_Y)
    weights = np.max(target_Y, axis=1) - np.min(target_Y, axis=1)
    if np.all(weights == 0):
        return None
    else:
        est_actions = np.argmax(predicted_Y, axis=1)
        target_actions = np.argmax(target_Y, axis=1)
        comparisons = est_actions == target_actions
        # encoding = np.where(comparisons, -1, 1) # where -1 is good (we are trying to minimize)
        encoding = np.where(comparisons, 0, 1) # where 0 is good (we are trying to minimize)
        return np.multiply(encoding, weights).sum()
        # return np.sqrt(np.multiply(np.power((predicted_Y - target_Y), 2), np.reshape(weights, (weights.shape[0], 1))).sum() / predicted_Y.shape[0])