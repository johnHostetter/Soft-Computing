#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:48:22 2021

@author: john
"""

import numpy as np

from scipy.special import softmax
from sklearn.metrics import mean_squared_error


def RMSE(predicted_Y, target_Y):
    """
    Calculates the Root-Mean-Square Error between the predicted output and the target output values.

    Parameters
    ----------
    predicted_Y : 2-D Numpy array
        The predicted values from a model.
    target_Y : 2-D Numpy array
        The target values for a model.

    Returns
    -------
    float
        Root-Mean-Square Error.

    """
    return np.sqrt(mean_squared_error(predicted_Y, target_Y))


def KL(predicted_Y, target_Y, tau=0.1):
    """
    Calculates the Kullback-Leibler Divergence between the predicted output and the target output values.

    Parameters
    ----------
    predicted_Y : 2-D Numpy array
        The predicted values from a model.
    target_Y : 2-D Numpy array
        The target values from a model.
    tau : float, optional
        The temperature of the softmax. The default is 0.1.

    Returns
    -------
    TYPE
        Kullback-Leibler Divergence.

    """
    return (softmax(target_Y / tau, axis=1) * (np.log(softmax(target_Y / tau, axis=1) / softmax(predicted_Y, axis=1)))).sum()