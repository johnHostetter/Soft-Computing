#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:50:21 2021

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def logistic(x, k, t='-'):
    """
    The logistic membership function.

    Parameters
    ----------
    x : float
        The input.
    k : float
        The bias (for logistic functions, k_i = v_i for all i in 1, 2, ... m).
    t : string, optional
        Adjusts whether this logistic membership function is describing term- or term+.
        The default is '-'.

    Returns
    -------
    float
        The degree of membership.

    """
    val = 2.0
    if t == '+':
        val = -2.0
    return 1.0 / (1.0 + np.exp(val * (x - k)))

def subs(x):
    """
    Substitutes True values for 1.0, and substitutes
    False values for -1.0. Necessary for the rules in
    the APFRB to calculate the consequents correctly.

    Parameters
    ----------
    x : boolean
        A boolean describing whether the linguistic term
        is term- or term+. If the linguistic term is term+,
        then x is True. Else, the linguistic term is term-,
        and x is False.

    Returns
    -------
    float
        A float value to modify the a_i values to have
        the correct sign when calculating a rule's consequent.

    """
    return 1.0 if x else -1.0

def foo(matrix, flc_rules):
    matrices = []
    ordered_rules = []
    copied_matrix = deepcopy(matrix)
    copied_flc_rules = deepcopy(flc_rules)
    copied_matrix = copied_matrix.astype('float32') 
    # for col_idx in range(matrix.shape[1]):
    col_idx = 0
    while col_idx < copied_matrix.shape[1]:
        if copied_matrix.shape[0] <= 2: # if the matrix is too small to further reduce
            matrices.append(copied_matrix)
            ordered_rules.append(copied_flc_rules)
            break
        
        # get column of interest and see how the antecedent is used across all rules
        col = copied_matrix[:,col_idx]
        nonzero_count = np.count_nonzero(col) - np.count_nonzero(np.isnan(col))
        
        condition_1 = nonzero_count == 1
        condition_2 = nonzero_count == (len(col) - 1)
        
        if condition_1 or condition_2:
            if condition_1:
                row_cond = ~np.isnan(col) & (col != 0)
            elif condition_2:
                row_cond = ~np.isnan(col) & (col == 0)
            row_idx = np.where(row_cond)[0][0]
            col_cond = ~np.isnan(col) & (col == 0)
            col = np.where(col_cond, np.nan, col)
            copied_matrix[:,col_idx] = col
            for idx, val in enumerate(copied_matrix[row_idx]):
                if idx != col_idx:
                    copied_matrix[row_idx, idx] = np.nan
            matrices.append(copied_matrix[row_idx])
            copied_matrix = np.vstack((copied_matrix[:row_idx], copied_matrix[row_idx+1:]))
            # keep track of the equivalent movements for flc rule base
            ordered_rules.append(copied_flc_rules[row_idx])
            copied_flc_rules.pop(row_idx)
            col_idx = 0 # reset the search
        else:
            col_idx += 1
    return matrices, ordered_rules

def foobar(unprocessed_tables, unprocesssed_hierarchical_flc_rules):
    from rule import FLC_Rule
    filtered_flc_rules = []
    for idx, item in enumerate(unprocesssed_hierarchical_flc_rules):
        corresponding_matrix = unprocessed_tables[idx]
        if isinstance(item, FLC_Rule):
            flc_rule = item
            keys = list(flc_rule.antecedents.keys())
            for jdx, val in enumerate(corresponding_matrix):
                if np.isnan(val):
                    flc_rule.antecedents[keys[jdx]] = None
            filtered_flc_rule_antecedents = {k: v for k, v in flc_rule.antecedents.items() if v is not None}
            filtered_flc_rules.append(FLC_Rule(filtered_flc_rule_antecedents, flc_rule.consequents))
        elif isinstance(item, list):
            temp = []
            for kdx in range(len(item)):
                flc_rule = item[kdx]
                keys = list(flc_rule.antecedents.keys())
                for jdx, val in enumerate(corresponding_matrix[kdx]):
                    if np.isnan(val):
                        flc_rule.antecedents[keys[jdx]] = None
                filtered_flc_rule_antecedents = {k: v for k, v in flc_rule.antecedents.items() if v is not None}
                temp.append(FLC_Rule(filtered_flc_rule_antecedents, flc_rule.consequents))
            filtered_flc_rules.append(temp)
        else:
            raise Exception('Invalid input. Something went wrong.')
    return filtered_flc_rules

def barfoo(ordered_matrices, ordered_flc_rules):
    from flc import FLC
    from rule import FLC_Rule
    copied_matrices = deepcopy(ordered_matrices)
    copied_flc_rules = deepcopy(ordered_flc_rules)
    next_idx = len(copied_flc_rules) - 1
    curr_idx = next_idx - 1
    else_clause = None

    while next_idx >= 0:
        next_table = copied_matrices[next_idx]
        next_item = copied_flc_rules[next_idx]
        if isinstance(next_item, list):
            else_clause = FLC(next_item, next_table)
        elif isinstance(next_item, FLC_Rule):
            next_item.else_clause = else_clause
            else_clause = next_item
        else:
            raise Exception('Invalid input. Something went wrong.')
        next_idx -= 1
        curr_idx -= 1
        
    return next_item

def barbar(hierarchical_rule):
    from flc import FLC
    rule = hierarchical_rule
    while not isinstance(rule, FLC):
        rule = rule.else_clause
        
    main_rule = rule.rules[0]
    idx = 1
    while True:
        flc_rule = rule.rules[idx]
        main_rule.else_clause = flc_rule
        main_rule = main_rule.else_clause
        rule.rules.pop(idx)
        if len(rule.rules) == 1:
            main_rule.default_class = True
            break
    return hierarchical_rule
            
# def foo(matrix):
#     try:
#         size = matrix.shape[1]
#         if size == 1:
#             raise IndexError
#         matrix = matrix.astype('float32') 
#         # for col_idx in range(size):
#         col_idx = 0
#         col = matrix[:,col_idx]
#         nonzero_count = np.count_nonzero(col)
        
#         if nonzero_count == 1:
#             row_cond = ~np.isnan(col) & (col != 0)
#             row_idx = np.where(row_cond)[0][0]
#             col_cond = ~np.isnan(col) & (col == 0)
#             col = np.where(col_cond, np.nan, col)
#             matrix[:,col_idx] = col
#             for idx, val in enumerate(matrix[row_idx]):
#                 if idx != col_idx:
#                     matrix[row_idx, idx] = np.nan
        
#         elif nonzero_count == len(col) - 1:
#             row_cond = ~np.isnan(col) & (col == 0)
#             row_idx = np.where(row_cond)[0][0]
#             col = np.where(row_cond, col, np.nan)
#             matrix[:,col_idx] = col
#             for idx, val in enumerate(matrix[row_idx]):
#                 if idx != col_idx:
#                     matrix[row_idx, idx] = np.nan
        
#         if matrix.shape[1] > 1 and (nonzero_count == 1 or nonzero_count == len(col) - 1):
#             row = matrix[row_idx]
#             arg = np.vstack((matrix[:row_idx], matrix[row_idx+1:]))
#             result = foo(arg)
#         else: # TODO: fix it so that this works when the first n columns cannot be reduced
#             arg = matrix[1:]
#             result = foo(arg)
#         try:
#             return (row, result)
#         except UnboundLocalError:
#             try:
#                 return result
#             except UnboundLocalError:
#                 return matrix
#     except IndexError:
#         return matrix.astype('float32') # matrix is a single vector

# def foo(matrix):
#     free_spot = 0
#     try:
#         size = matrix.shape[1]
#         if size == 1:
#             raise IndexError
#         matrix = matrix.astype('float32') 
#         for col_idx in range(size):
#             col = matrix[:,col_idx]
#             nonzero_count = np.count_nonzero(col)
            
#             if nonzero_count == 1:
#                 row_cond = ~np.isnan(col) & (col != 0)
#                 row_idx = np.where(row_cond)[0][0]
#                 col_cond = ~np.isnan(col) & (col == 0)
#                 col = np.where(col_cond, np.nan, col)
#                 matrix[:,col_idx] = col
#                 for idx, val in enumerate(matrix[row_idx]):
#                     if idx != col_idx:
#                         matrix[row_idx, idx] = np.nan
            
#             elif nonzero_count == len(col) - 1:
#                 row_cond = ~np.isnan(col) & (col == 0)
#                 row_idx = np.where(row_cond)[0][0]
#                 col = np.where(row_cond, col, np.nan)
#                 matrix[:,col_idx] = col
#                 for idx, val in enumerate(matrix[row_idx]):
#                     if idx != col_idx:
#                         matrix[row_idx, idx] = np.nan
            
#         return matrix
                    
#     except IndexError:
#         return matrix.astype('float32') # matrix is a single vector

# def foo(matrix):
#     rows_moved_record = {'from':None, 'to':None} # keep a record of any row moved
#     free_spot = 0
#     try:
#         size = matrix.shape[1]
#         if size == 1:
#             raise IndexError
#         matrix = matrix.astype('float32') 
#         # for col_idx in range(size):
#         col_idx = 0
#         col = matrix[:,col_idx]
#         nonzero_count = np.count_nonzero(col)
        
#         if nonzero_count == 1:
#             row_cond = ~np.isnan(col) & (col != 0)
#             row_idx = np.where(row_cond)[0][0]
#             col_cond = ~np.isnan(col) & (col == 0)
#             col = np.where(col_cond, np.nan, col)
#             matrix[:,col_idx] = col
#             for idx, val in enumerate(matrix[row_idx]):
#                 if idx != col_idx:
#                     matrix[row_idx, idx] = np.nan
#             # the rule must be made to be the first rule on the top
#             temp = matrix.tolist()
#             row_to_be_added = temp[row_idx]
#             temp.pop(row_idx)
#             temp.insert(free_spot, row_to_be_added)
#             rows_moved_record['to'] = free_spot
#             rows_moved_record['from'] = row_idx
#             free_spot += 1
#             matrix = np.array(temp)
        
#         elif nonzero_count == len(col) - 1:
#             row_cond = ~np.isnan(col) & (col == 0)
#             row_idx = np.where(row_cond)[0][0]
#             col = np.where(row_cond, col, np.nan)
#             matrix[:,col_idx] = col
#             for idx, val in enumerate(matrix[row_idx]):
#                 if idx != col_idx:
#                     matrix[row_idx, idx] = np.nan
#             # the rule must be made to be the first rule on the top
#             temp = matrix.tolist()
#             row_to_be_added = temp[row_idx]
#             temp.pop(row_idx)
#             temp.insert(free_spot, row_to_be_added)
#             rows_moved_record['to'] = free_spot
#             rows_moved_record['from'] = row_idx
#             free_spot += 1
#             matrix = np.array(temp)
        
#         if matrix.shape[1] > 1:
#             arg = matrix[1:, 1:]
#             # col_to_be_added_back = matrix[:, 0]
#             row_to_be_added_back = matrix[0, 1:]
#             matrix = matrix[:,:1]
#             result, move_record = foo(arg)
            
#             if move_record['to'] is not None and move_record['from'] is not None:
#                 temp = matrix.tolist()
                
            
#             num_cols_to_be_padded = result.shape[1]
#             # padding = np.array([np.nan]*num_cols_to_be_padded).reshape((1, num_cols_to_be_padded))
#             result = np.vstack((row_to_be_added_back, result))
#             matrix = np.hstack((matrix, result))
#         return matrix, rows_moved_record
                
#     except IndexError:
#         return matrix.astype('float32') # matrix is a single vector

# inpt = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0]])

# results1 = foo(inpt)

# test = np.array([[1, 0], [1, 1], [0, 0]])

# ressults2 = foo(test)
        
def plot(title, x_lbl, y_lbl):
    """
    Handles the basic mechanics of plotting a graph
    such as assigning the x or y label, title, etc.
    Shows the plot after the function call is complete.

    Parameters
    ----------
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.title(title)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()

def line(x, y, title, x_lbl, y_lbl):
    """


    Parameters
    ----------
    x : list
        A list containing the x values.
    y : list
        A list containing the y values.
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.plot(x, y)
    plot(title, x_lbl, y_lbl)

def bar(x, heights, title, x_lbl, y_lbl):
    """


    Parameters
    ----------
    x : list
        A list containing the x values.
    heights : list
        A list containing the heights of the bars.
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.bar(x, heights)
    plot(title, x_lbl, y_lbl)
