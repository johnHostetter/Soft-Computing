#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:50:11 2021

@author: john
"""

import os
import sys
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

class DirectoriesContextManager(object):
    # https://stackoverflow.com/questions/17211078/how-to-temporarily-modify-sys-path-in-python 
    def __init__(self, path):
        self.path = path
    
    def __enter__(self):
        try:
            # ignore any directory that has '.' in it (e.g. .gitignore)
            self.directories = [folder for folder in os.listdir(self.path) if '.' not in folder]
            
            for directory in self.directories:
                if directory not in sys.path:
                    sys.path.append(self.path + '/' + directory)
        except FileNotFoundError:
            pass
        
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for directory in self.directories:
                if directory not in sys.path:
                    sys.path.remove(self.path + '/' + directory)
        except ValueError:
            pass