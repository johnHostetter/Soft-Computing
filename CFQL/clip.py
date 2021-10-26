#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:21:50 2021

@author: john
"""

import time
import numpy as np

def gaussian(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))

def R(sigma_1, sigma_2):
    # regulator function
    return (1/2) * (sigma_1 + sigma_2)

def CLIP(X, Y, mins, maxes, terms=[], alpha=0.2, beta=0.6, theta=1e-8):
    # theta is a parameter I add to accomodate for the instance in which an observation has values that are the minimum/maximum
    # otherwise, when determining the Gaussian membership, a division by zero will occur
    # it essentially acts as an error tolerance
    antecedents = terms
    min_values_per_feature_in_X = mins
    max_values_per_feature_in_X = maxes
    for idx, training_tuple in enumerate(zip(X, Y)):
        x = training_tuple[0]
        d = training_tuple[1]
        if not antecedents:
            # no fuzzy clusters yet, create the first fuzzy cluster
            for p in range(len(x)):
                c_1p = x[p]
                min_p = min_values_per_feature_in_X[p]
                max_p = max_values_per_feature_in_X[p]
                left_width = np.sqrt(-1.0 * (np.power((min_p - x[p]) + theta, 2) / np.log(alpha)))
                right_width = np.sqrt(-1.0 * (np.power((max_p - x[p]) + theta, 2) / np.log(alpha)))
                sigma_1p = R(left_width, right_width)
                antecedents.append([{'center': c_1p, 'sigma': sigma_1p, 'support':1}])
        else:
            # calculate the similarity between the input and existing fuzzy clusters
            for p in range(len(x)):
                SM_jps = []
                for j, A_jp in enumerate(antecedents[p]):
                    SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                    SM_jps.append(SM_jp)
                j_star_p = np.argmax(SM_jps)

                if np.max(SM_jps) > beta:
                    # the best matched cluster is deemed as being able to give satisfactory description of the presented value
                    A_j_star_p = antecedents[p][j_star_p]
                    A_j_star_p['support'] += 1
                else:
                    # a new cluster is created in the input dimension based on the presented value                    
                    jL_p = None
                    jR_p = None
                    jL_p_differences = []
                    jR_p_differences = []
                    for j, A_jp in enumerate(antecedents[p]):
                        c_jp = A_jp['center']
                        if c_jp >= x[p]:
                            continue # the newly created cluster has no immediate left neighbor
                        else:
                            jL_p_differences.append(np.abs(c_jp - x[p]))
                    try:
                        jL_p = np.argmin(jL_p_differences)
                    except ValueError:
                        jL_p = None
                        
                    for j, A_jp in enumerate(antecedents[p]):
                        c_jp = A_jp['center']
                        if c_jp <= x[p]:
                            continue # the newly created cluster has no immediate right neighbor
                        else:
                            jR_p_differences.append(np.abs(c_jp - x[p]))
                    try:
                        jR_p = np.argmin(jR_p_differences)
                    except ValueError:
                        jR_p = None
                    
                    new_c = x[p]
                    new_sigma = None
                    
                    if jL_p is None and jR_p is None:
                        continue
                    
                    if jL_p is None:
                        cR_jp = antecedents[p][jR_p]['center']
                        sigma_R_jp = antecedents[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(alpha)))
                        sigma_R = R(left_sigma_R, sigma_R_jp)
                        
                        new_sigma = sigma_R
                    elif jR_p is None:
                        cL_jp = antecedents[p][jL_p]['center']
                        sigma_L_jp = antecedents[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(alpha)))
                        sigma_L = R(left_sigma_L, sigma_L_jp)
                        
                        new_sigma = sigma_L
                    else:
                        cR_jp = antecedents[p][jR_p]['center']
                        sigma_R_jp = antecedents[p][jR_p]['sigma']
                        left_sigma_R = np.sqrt(-1.0 * (np.power(cR_jp - x[p], 2) / np.log(alpha)))
                        sigma_R = R(left_sigma_R, sigma_R_jp)
                        
                        cL_jp = antecedents[p][jL_p]['center']
                        sigma_L_jp = antecedents[p][jL_p]['sigma']
                        left_sigma_L = np.sqrt(-1.0 * (np.power(cL_jp - x[p], 2) / np.log(alpha)))
                        sigma_L = R(left_sigma_L, sigma_L_jp)
                        
                        new_sigma = R(sigma_R, sigma_L)
                    antecedents[p].append({'center':new_c, 'sigma':new_sigma, 'support':1})
    return antecedents

def rule_creation(X, Y, antecedents, consequents, existing_rules=[], existing_weights=[], problem_type='SL'):
    start = time.time()
    rules = existing_rules
    weights = existing_weights
    for training_tuple in zip(X, Y):
        x = training_tuple[0]
        d = training_tuple[1]
        
        CF = 1.0 # certainty factor of this rule
        A_star_js = []
        for p in range(len(x)):
            SM_jps = []
            for j, A_jp in enumerate(antecedents[p]):
                SM_jp = gaussian(x[p], A_jp['center'], A_jp['sigma'])
                SM_jps.append(SM_jp)
            CF *= np.max(SM_jps)
            j_star_p = np.argmax(SM_jps)
            A_star_js.append(j_star_p)
            
        if problem_type == 'SL':
            C_star_qs = []
            for q in range(len(d)):
                SM_jqs = []
                for j, C_jq in enumerate(consequents[q]):
                    SM_jq = gaussian(d[q], C_jq['center'], C_jq['sigma'])
                    SM_jqs.append(SM_jq)
                CF *= np.max(SM_jqs)
                j_star_q = np.argmax(SM_jqs)
                C_star_qs.append(j_star_q)
            
            R_star = {'A':A_star_js, 'C': C_star_qs, 'CF': CF, 'time_added': start}
            
        elif problem_type == 'RL':       
            C_star_qs = []
            for q in range(len(d)):
                SM_jqs = []
                for j, C_jq in enumerate(consequents[q]):
                    SM_jq = gaussian(d[q], C_jq['center'], C_jq['sigma'])
                    SM_jqs.append(SM_jq)
                CF *= np.max(SM_jqs)
                j_star_q = np.argmax(SM_jqs)
                from copy import deepcopy
                new_consequent = deepcopy(consequents[q][j_star_q])
                new_consequent['sigma'] = 1.0
                consequents[q].append(new_consequent)
                new_consequent_index = len(consequents[q]) - 1
                C_star_qs.append(new_consequent_index)
            # from copy import deepcopy
            # C_star_qs = []
            # for q in range(Y.shape[1]):
            #     import random
            #     new_consequent = {'center':0.0, 'sigma':1.0, 'support':1}
            #     consequents[q].append(new_consequent)
            #     new_consequent_index = len(consequents[q]) - 1
            #     C_star_qs.append(new_consequent_index)
            R_star = {'A':A_star_js, 'C':C_star_qs, 'CF': CF, 'time_added': start}
        
        if not rules:
            # no rules in knowledge base yet
            rules.append(R_star)
            weights.append(1.0)
        else:
            # check for uniqueness
            add_new_rule = True
            for k, rule in enumerate(rules):
                try:
                    if (rule['A'] == R_star['A']) and (rule['C'] == R_star['C']):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
                    elif (rule['A'] == R_star['A']): # my own custom else-if statement
                        if problem_type == 'RL':
                            add_new_rule = False
                        elif rule['CF'] <= R_star['CF']:
                            add_new_rule = False
                except ValueError: # this happens because R_star['A'] and R_star['C'] are Numpy arrays
                    if all(rule['A'] == list(R_star['A'])) and all(rule['C'] == list(R_star['C'])):
                        # the generated rule is not unique, it already exists, enhance this rule's weight
                        weights[k] += 1.0
                        rule['CF'] = min(rule['CF'], R_star['CF'])
                        add_new_rule = False
                        break
                    elif all(rule['A'] == list(R_star['A'])): # my own custom else-if statement
                        if rule['CF'] <= R_star['CF']:
                            add_new_rule = False
            if add_new_rule:
                rules.append(R_star)
                weights.append(1.0)
                
    # check for consistency
    all_antecedents = [rule['A'] for rule in rules]

    repeated_rule_indices = set()
    for k in range(len(rules)):
        indices = np.where(np.all(all_antecedents == np.array(rules[k]['A']), axis=1))[0]
        if len(indices) > 1: 
            if len(repeated_rule_indices) == 0: # this can be combined with the following elif-statement
                repeated_rule_indices.add(tuple(indices))
            elif len(repeated_rule_indices) > 0: # this can be combined with the above if-statement
                repeated_rule_indices.add(tuple(indices))
    
    for indices in repeated_rule_indices:
        # weights_to_compare = [rules[idx]['CF'] for idx in indices] # HyFIS approach to rule creation
        weights_to_compare = [weights[idx] for idx in indices]
        strongest_rule_index = indices[np.argmax(weights_to_compare)] # keep the rule with the greatest weight to it
        for index in indices:
            if index != strongest_rule_index:
                rules[index] = None
                weights[index] = None
    rules = [rules[k] for k, rule in enumerate(rules) if rules[k] is not None]
    weights = [weights[k] for k, weight in enumerate(weights) if weights[k] is not None]

    # need to check that no antecedent/consequent terms are "orphaned"
    
    all_antecedents = [rule['A'] for rule in rules]
    all_antecedents = np.array(all_antecedents)
    for p in range(len(x)):
        if len(antecedents[p]) == len(np.unique(all_antecedents[:,p])):
            continue
        else:
            # orphaned antecedent term
            indices_for_antecedents_that_are_used = set(all_antecedents[:,p])
            updated_indices_to_map_to = list(range(len(indices_for_antecedents_that_are_used)))
            antecedents[p] = [antecedents[p][index] for index in indices_for_antecedents_that_are_used]
            
            paired_indices = list(zip(indices_for_antecedents_that_are_used, updated_indices_to_map_to))
            for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                original_index = index_pair[0] # so, when we updated the original index to its new index
                new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                all_antecedents[:,p][all_antecedents[:,p] == original_index] = new_index
            
    all_consequents = [rule['C'] for rule in rules]
    all_consequents = np.array(all_consequents)
    for q in range(len(d)):
        if len(consequents[q]) == len(np.unique(all_consequents[:,q])):
            continue
        else:
            # orphaned consequent term
            indices_for_consequents_that_are_used = set(all_consequents[:,q])
            updated_indices_to_map_to = list(range(len(indices_for_consequents_that_are_used)))
            consequents[q] = [consequents[q][index] for index in indices_for_consequents_that_are_used]
            
            paired_indices = list(zip(indices_for_consequents_that_are_used, updated_indices_to_map_to))
            for index_pair in paired_indices: # the paired indices are sorted w.r.t. the original indices
                original_index = index_pair[0] # so, when we updated the original index to its new index
                new_index = index_pair[1] # we are guaranteed not to overwrite the last updated index
                all_consequents[:,q][all_consequents[:,q] == original_index] = new_index
                
    # update the rules in case any orphaned terms occurred
    for idx, rule in enumerate(rules):
        rule['A'] = all_antecedents[idx]
        rule['C'] = all_consequents[idx]

    return antecedents, consequents, rules, weights