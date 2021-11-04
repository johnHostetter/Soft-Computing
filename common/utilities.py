#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:50:11 2021

@author: john
"""

import numpy as np

def boolean_indexing(v, fillval=np.nan):
    """
    Converts uneven list of lists to Numpy array with np.nan as padding for smaller lists.
    
    https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with

    Parameters
    ----------
    v : list
        A list containing lists that have an uneven number of elements in each list.
    fillval : float, optional
        The value to pad each list with if it is 'missing' elements. The default is np.nan.

    Returns
    -------
    out : 2-D Numpy array
        A 2-D Numpy array representation of the list of lists, but with 'fillval' as the padding at the end of each row.

    """
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out