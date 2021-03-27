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
            val_to_keep = col[row_idx]
            col_cond = ~np.isnan(col) & (col != val_to_keep)
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
            if not col_idx < copied_matrix.shape[1]:
                matrices.append(copied_matrix)
                ordered_rules.append(copied_flc_rules)
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
            if len(ordered_flc_rules) == 1: # the fuzzy logic controller is the only entry 
                return else_clause
        elif isinstance(next_item, FLC_Rule):
            next_item.else_clause = else_clause
            else_clause = next_item
        else:
            raise Exception('Invalid input. Something went wrong.')
        next_idx -= 1
        curr_idx -= 1
        
    return next_item

def barbar(hierarchical_rule, default):
    from flc import FLC
    rule = hierarchical_rule
    while not isinstance(rule, FLC):
        rule.ordinary_logic = True
        rule = rule.else_clause
        
    main_rule = rule.rules[0]
    idx = 1
    while True:
        if len(rule.rules) == 1:
            # main_rule.default_class = True
            main_rule.ordinary_logic = True
            main_rule.else_clause = default
            # we no longer need the fuzzy logic controller class, there is only one rule left
            return hierarchical_rule
            break
        flc_rule = rule.rules[idx]
        flc_rule.ordinary_logic = True
        main_rule.else_clause = flc_rule
        main_rule.ordinary_logic = True
        main_rule = main_rule.else_clause
        rule.rules.pop(idx)
    return hierarchical_rule

# def barbar(hierarchical_rule):
#     from flc import FLC
#     rule = hierarchical_rule
#     while not isinstance(rule, FLC):
#         rule.ordinary_logic = True
#         rule = rule.else_clause
        
#     main_rule = rule.rules[0]
#     idx = 1
#     while True:
#         flc_rule = rule.rules[idx]
#         flc_rule.ordinary_logic = True
#         main_rule.else_clause = flc_rule
#         main_rule.ordinary_logic = True
#         main_rule = main_rule.else_clause
#         rule.rules.pop(idx)
#         if len(rule.rules) == 1:
#             main_rule.default_class = True
#             main_rule.ordinary_logic = True
#             break
#     return hierarchical_rule

def default_consequent(ordered_table, filtered_rules):
    from flc import FLC
    from rule import FLC_Rule
    consequent_frequency = {} # find the frequency for each rule's consequent term
    for item in filtered_rules:
        if isinstance(item, FLC_Rule):
            consequent = item.consequent()
            if consequent in consequent_frequency:
                consequent_frequency[consequent] += 0
            else:
                consequent_frequency[consequent] = 0
        elif isinstance(item, list):
            for flc_rule in item:
                consequent = flc_rule.consequent()
                if consequent in consequent_frequency:
                    consequent_frequency[consequent] += 1
                else:
                    consequent_frequency[consequent] = 1
        else:
            raise Exception('Invalid input. Something went wrong.')
    # return the dictionary key that has the maximum value
    # WARNING: will only return 1 of many matches (if there is a tie), however, this is okay for this purpose
    return max(consequent_frequency, key=lambda k: consequent_frequency[k])

def delete_rules_with_default_consequent(ordered_table, filtered_rules):
    from rule import FLC_Rule
    default = default_consequent(ordered_table, filtered_rules)
    indices_to_delete = []
    for idx, item in enumerate(filtered_rules):
        if isinstance(item, FLC_Rule):
            if item.consequent() == default:
                indices_to_delete.append(idx)
        elif isinstance(item, list):
            for jdx, flc_rule in enumerate(item):
                if flc_rule.consequent() == default:
                    indices_to_delete.append((idx, jdx))
        else:
            raise Exception('Invalid input. Something went wrong.')
    
    # delete all identified ruless
    indices_to_delete.reverse() # reverse the list to not affect the indices that are to be deleted as the elements are removed from the list
    for index in indices_to_delete:
        if isinstance(index, int):
            ordered_table.pop(index)
            filtered_rules.pop(index)
        elif isinstance(index, tuple):
            ordered_table = np.delete(ordered_table[index[0]], index[1], axis=1)
            filtered_rules[index[0]].pop(index[1])
        else:
            raise Exception('Invalid input. Something went wrong.')
    return ordered_table, filtered_rules, default

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

PROBLEM_FEATURES = ['ntellsSinceElicit', 'ntellsSinceElicitKC', 'nElicitSinceTell', 'nElicitSinceTellKC', 'pctElicit',
			   'pctElicitKC', 'pctElicitSession', 'pctElicitKCSession', 'nTellSession', 'nTellKCSession', 'kcOrdering',
			   'kcOrderingSession', 'durationKCBetweenDecision', 'timeInSession', 'timeBetweenSession',
			   'timeOnTutoring',
			   'timeOnTutoringWE', 'timeOnTutoringPS', 'timeOnTutoringKC', 'timeOnTutoringKCWE', 'timeOnTutoringKCPS',
			   'avgTimeOnStep', 'avgTimeOnStepWE', 'avgTimeOnStepPS', 'avgTimeOnStepKC', 'avgTimeOnStepKCWE',
			   'avgTimeOnStepKCPS', 'timeOnLastStepKCPS', 'timeOnTutoringSession', 'timeOnTutoringSessionWE',
			   'timeOnTutoringSessionPS', 'avgTimeOnStepSession', 'avgTimeOnStepSessionWE', 'avgTimeOnStepSessionPS',
			   'nTotalHint', 'nTotalHintSession', 'nHintKC', 'nHintSessionKC', 'AvgTimeOnStepWithHint',
			   'durationSinceLastHint', 'stepsSinceLastHint', 'stepsSinceLastHintKC', 'totalTimeStepsHint',
			   'totalStepsHint',
			   'earlyTraining', 'simpleProblem', 'nKCs', 'nKCsAsPS', 'nKCsSession', 'nKCsSessionPS',
			   'newLevelDifficulty',
			   'nPrincipleInProblem', 'quantitativeDegree', 'nTutorConceptsSession', 'tutAverageConcepts',
			   'tutAverageConceptsSession', 'tutConceptsToWords', 'tutConceptsToWordsSession', 'tutAverageWords',
			   'tutAverageWordsSession', 'tutAverageWordsPS', 'tutAverageWordsSessionPS', 'nDistinctPrincipleInSession',
			   'nPrincipleInSession', 'problemDifficuly', 'problemCompexity', 'problemCategory', 'pctCorrect',
			   'pctOverallCorrect', 'nCorrectKC', 'nIncorrectKC', 'pctCorrectKC', 'pctOverallCorrectKC',
			   'nCorrectKCSession',
			   'nIncorrectKCSession', 'pctCorrectSession', 'pctCorrectKCSession', 'pctOverallCorrectSession',
			   'pctOverallCorrectKCSession', 'nStepSinceLastWrong', 'nStepSinceLastWrongKC', 'nWEStepSinceLastWrong',
			   'nWEStepSinceLastWrongKC', 'nStepSinceLastWrongSession', 'nStepSinceLastWrongKCSession',
			   'nWEStepSinceLastWrongSession', 'nWEStepSinceLastWrongKCSession', 'timeSinceLastWrongStepKC',
			   'nCorrectPSStepSinceLastWrong', 'nCorrectPSStepSinceLastWrongKC',
			   'nCorrectPSStepSinceLastWrongKCSession',
			   'pctCorrectPrin', 'pctCorrectPrinSession', 'nStepSinceLastWrongPrin', 'nWEStepSinceLastWrongPrin',
			   'nStepSinceLastWrongPrinSession', 'nWEStepSinceLastWrongPrinSession', 'nCorrectPSStepSinceLastWrongPrin',
			   'nCorrectPSStepSinceLastWrongPrinSession', 'pctCorrectFirst', 'nStepsSinceLastWrongFirst',
			   'nWEStepSinceLastWrongFirst', 'nCorrectPSStepSinceLastWrongFirst', 'pctCorrectLastProb',
			   'pctCorrectLastProbPrin', 'pctCorrectAdd2Select', 'pctCorrectAdd3Select', 'pctCorrectCompSelect',
			   'pctCorrectDeMorSelect', 'pctCorrectIndeSelect', 'pctCorrectMutualSelect', 'pctCorrectAdd2Apply',
			   'pctCorrectAdd3Apply', 'pctCorrectCompApply', 'pctCorrectDeMorApply', 'pctCorrectIndeApply',
			   'pctCorrectMutualApply', 'pctCorrectAdd2All', 'pctCorrectAdd3All', 'pctCorrectCompAll',
			   'pctCorrectDeMorAll',
			   'pctCorrectIndeAll', 'pctCorrectMutualAll', 'pctCorrectSelectMain', 'nAdd2Prob', 'nAdd3Prob',
			   'nDeMorProb',
			   'nIndeProb', 'nCompProb', 'nMutualProb']