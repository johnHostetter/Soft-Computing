#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:41:25 2021

@author: john
"""

import numpy as np

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

def boolean_indexing(v, fillval=np.nan):
    """
    Converts uneven list of lists to Numpy array with np.nan as padding for smaller lists.
    
    https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    fillval : TYPE, optional
        DESCRIPTION. The default is np.nan.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out