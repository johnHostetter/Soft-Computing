#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:54:08 2021

@author: john
"""

import time
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

try:
    from .flc import FLC
    from .common import foo, foobar, barfoo, barbar, bar, line
    from .transformation import APFRB_rule_to_FLC_rule
except ImportError:
    from flc import FLC
    from common import foo, foobar, barfoo, barbar, bar, line
    from transformation import APFRB_rule_to_FLC_rule

class RuleReducer:
    def __init__(self, apfrb):
        self.apfrb = apfrb
        self.flc = None
        
    def step_1(self, skip=True):
        """
        For each k, if the abs(a_k) is small,
        remove the atoms containing x_k in the IF part,
        and remove a_k from the THEN part of all the rules

        Parameters
        ----------
        skip : bool, optional
            Controls whether to enable this step in the simplification process. The default is True.

        Returns
        -------
        None.

        """
        if not skip:
            print('\nStep 1 in progress (this requires your attention)...')
            yes = True
            print('\nWould you like to remove an antecedent from the IF part? [y/n]')
            yes = input().lower() == 'y'
            while(yes):
                print('\nStep 1 in progress (this should be quick)...')
                sorted_a = sorted([abs(x) for x in self.apfrb.a[1:]]) # ignore the output node bias, find absolute values, and sort
                bar(range(len(sorted_a)), sorted_a, 'The values of a_k', 'The size of vector a (except a_0)', 'The value of a_i (where 0 < i <= m)')
    
                try:
                    print('\nHow many of the smallest values would you like to retrieve? [type \'cancel\' to skip Step 1]')
                    raw = input()
                    ans = int(raw)
                    print('\n%s' % sorted_a[:ans])
                except Exception:
                    if raw.lower() == 'cancel':
                        print('\nSkipping step 1.')
                        break
                    else:
                        print('\nInvalid response. Unsure of how to respond. Resetting step 1.')
                        continue
    
                try:
                    print('\nUp to and including which value\'s index would you like to remove until? [type \'cancel\' to skip Step 1]')
                    raw = input()
                    ans = int(raw)
                    small_val = sorted_a[ans]
                    temp = np.array([abs(x) for x in self.a[1:]])
                    small_val_indices = np.where(temp <= small_val)[0]
                    print('\nDeleting up to, and including, a_i = %s...' % small_val)
                except Exception:
                    if raw.lower() == 'cancel':
                        print('\nSkipping step 1.')
                        break
                    else:
                        print('\nInvalid response. Unsure of how to respond. Resetting step 1.')
                        continue
    
                num_of_rules_to_delete = len(self.rules) - (len(self.rules) / (2 * len(small_val_indices)))
                print('\nConfirm the deletion of %s fuzzy logic rules (out of %s rules). [y/n]' % (num_of_rules_to_delete, len(self.rules)))
                delete = input().lower() == 'y'
    
                if delete:
                    while small_val_indices.any():
                        index = small_val_indices[0]
                        self.__delete(index)
                        temp = np.array([abs(x) for x in self.a[1:]])
                        small_val_indices = np.where(temp <= small_val)[0]
    
                print('\nThe All Permutations Rule Base now has %s rules.' % len(self.rules))
                print('\nWould you like to remove another antecedent from the IF part? [y/n]')
                yes = input().lower() == 'y'
        else:
            print('\nSkipping step 1...')
                
    def entries_to_csv(self, entries, i, j):
        df = pd.DataFrame(entries)
        file_name = './data/rules_{}_to_{}.csv'.format(i, j)
        df.to_csv(file_name, index=False)
        return {'rule':[], 'z':[], 't_k_z':[], 'c_k_z':[], 'max(m_k)*max(c_k)':[]} 
    
    def determine_rule_activations(self, rules, data):
        print('\nDetermine rule activations')
        if isinstance(rules, dict):
            rule_labels = list(rules.keys())
            rules = list(rules.values())
            multiprocessing = True # rules object is dictionary only when using multiprocessing
        else:
            rule_labels = [*range(len(rules))]
            multiprocessing = False
        
        entries = {'rule':[], 'z':[], 't_k_z':[], 'c_k_z':[], 'max(m_k)*max(c_k)':[]}    
        
        start_time = time.time()
        m_k_l_ks = []
        q = len(rules)
        for k in range(q):
            if k == q / 4:
                current_time = time.time()
                print('\nA quarter of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
                if multiprocessing:
                    entries = self.entries_to_csv(entries, rule_labels[0], rule_labels[k-1])
            elif k == q / 2:
                current_time = time.time()
                print('\nHalfway done [elapsed time: %s seconds]...' % (current_time - start_time))
                if multiprocessing:
                    entries = self.entries_to_csv(entries, rule_labels[int(q/4)], rule_labels[k-1])
            elif k == 3 * q / 4:
                current_time = time.time()
                print('\nThree quarters of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
                if multiprocessing:
                    entries = self.entries_to_csv(entries, rule_labels[int(q/2)], rule_labels[k-1])
            t_ks = []
            c_ks = []
            rule_k = rules[k]
            z_i = 0
            for z in data:
                t_ks.append(rule_k.t(z))
                c_ks.append(self.apfrb.c_k(z, k))
                entries['rule'].append(rule_labels[k])
                entries['z'].append(z_i)
                entries['t_k_z'].append(t_ks[-1])
                entries['c_k_z'].append(c_ks[-1])
                z_i += 1
            
            m_k = max(t_ks)
            l_k = max(c_ks)
            m_k_l_ks.append(m_k * l_k)
            entries['max(m_k)*max(c_k)'].extend([m_k_l_ks[-1]] * (z_i))
        if multiprocessing:
            _ = self.entries_to_csv(entries, rule_labels[3 * int(q/4)], rule_labels[q - 1])
        else:
            _ = self.entries_to_csv(entries, 0, (len(rules)-1))
        return m_k_l_ks
    
    def __delete_rules_by_indices(self, indices):
        # iterate through the rule base, swapping out rules with NoneType to delete later
        for rule_index in indices:
            self.apfrb.rules[rule_index] = None
            self.apfrb.table[rule_index] = None
        while True:
            try:
                self.apfrb.rules.remove(None)
                self.apfrb.table.remove(None)
            except Exception: # Exception is raised when there is no more NoneType objects to remove
                self.apfrb.r = len(self.apfrb.rules) # update the stored count of number of rules
                break
    
    def step_2(self, Z, MULTIPROCESSING=False, PROCESSES=2, skip=False):
        """
        For each rule k, compute m_k and l_k,
        if m_k * l_k is small, then delete rule k from APFRB
        (WARNING: this results in a Fuzzy Logic Controller)

        Parameters
        ----------
        Z : Numpy 2-D array.
            Raw data observations.
        MULTIPROCESSING : bool, optional
            Enables multiprocessing. The default is False.
        PROCESSES : int, optional
            Determines the number of processes for multiprocessing. The default is 2.
        skip : bool, optional
            Controls whether to enable this step in the simplification process. The default is False.

        Returns
        -------
        None.

        """
        if not skip:
            print('\nStep 2 in progress (this might take awhile)...')
            m_k_l_ks = []
    
            if MULTIPROCESSING:
                with Pool(PROCESSES) as p:
                    rule_activations = partial(self.determine_rule_activations, data=Z)
                    rules_list = np.array_split(self.apfrb.rules, PROCESSES)
                    ite = 1
                    rules_dict_list = []
                    for rule_list in rules_list:
                        rules_dict_list.append(dict(zip(range(ite, ite + len(rule_list)), rule_list)))
                        ite += (len(rule_list))
                    m_k_l_ks = p.map(rule_activations, rules_dict_list)
                    m_k_l_ks = list(itertools.chain(*m_k_l_ks))
            else:
                m_k_l_ks = self.determine_rule_activations(self.apfrb.rules, data=Z)
    
            # x coordinate is the number of rules, y coordinate is m_k * l_k
            q = len(m_k_l_ks)
            xs = range(q)
            ys = sorted(m_k_l_ks)
            line(range(q), sorted(m_k_l_ks), 'The m_k * l_k of each rule', 'Rules', 'm_k * l_k')
    
        print('\nThe five smallest m_k * l_k values: \n\n%s' % sorted(m_k_l_ks)[:5])

        m = np.diff(ys) / np.diff(xs)
        
        idx = np.nonzero(np.round(np.diff(m), 1))[0][0] + 1
        epsilon = ys[idx]
        indices_of_rules_to_delete = np.where(np.array(m_k_l_ks) <= epsilon)[0]
        print('\nThere are %s fuzzy logic rules that will be deleted.' % len(indices_of_rules_to_delete))
        self.__delete_rules_by_indices(indices_of_rules_to_delete)
        
    def step_3(self, Z, skip=True):
        """
        Determines whether Mean of Maximum defuzzification can be used.
        To answer this, calculate e/r, and if e/r is small, then output f_k(x) instead of f(x).
        
        CAUTION: This step is rarely/never used, as e/r is often not small enough to 
        use MoM defuzzification. Therefore, it is not sufficiently tested, and might not work.

        Parameters
        ----------
        skip : bool, optional
            Controls whether to enable this step in the simplification process. The default is True.

        Returns
        -------
        None.

        """
        # determines whether Mean of Maximum defuzzification can be used, if e / r is small enough
        if False:
            k = None
            x = None
            e_rs = []

            # NOTE: any x in the training set, D, can be used. Therefore, any arbitrary x can
            # be selected, and the for loop immediately below is not required.
            for x in Z:
                xs = []
                t_ks = []
                for i in range(len(self.rules)):
                    rule_i = self.rules[i]
                    t_ks.append(rule_i.t(x))
                    xs.append(x)

                # k(x) = argmax_i t_i(x)
                k = t_ks.index(max(t_ks))
                x_ = xs[k]

                # compute e
                bs = []
                for x in Z:
                    bs.append(self.__b(x, k))
                e = max(bs)

                # compute r
                vals = []
                for x in Z:
                    t_k = self.rules[k].t(x)
                    denominator = 0.0
                    for i in range(len(self.rules)):
                        if i != k:
                            denominator += self.rules[i].t(x)
                    vals.append(t_k / denominator)
                r = 1.0 + min(vals)
                e_rs.append(e/r)
                
    def step_4(self):
        """
        If a specific atom (e.g. "x_1 is smaller than 7")
        appears in all the rules, then delete it from all of them

        Returns
        -------
        FLC
            An ordinary fuzzy logic controller.

        """
        # beyond this point, inference no longer works
        # TODO: fix fuzzy logic inference
        flc_rules = []
        for apfrb_rule in self.apfrb.rules:
            flc_rules.append(APFRB_rule_to_FLC_rule(apfrb_rule))

        # step 4
        table = deepcopy(np.where(self.apfrb.table, 1, 0))
        table = table.astype('float32')
        # table = np.matrix(self.apfrb.table) # TODO: update self.table so it is consistent with the new table
        # for i in range(len(self.table[0])):
        
        # identify the antecedents to be deleted by traversing the table
        antecedents_to_delete = []
        for col_idx in range(table.shape[1]):
            col = table[:,col_idx]
            # either all elements in the column are 1 or all elements in the column are 0
            if sum(col) == table.shape[0] or sum(col) == 0:
                antecedents_to_delete.append(col_idx + 1) # plus 1 since the antecedents begin count from '1'
        
        # delete the antecedents from the fuzzy logic control rules
        for flc_rule in flc_rules:
            for key in antecedents_to_delete:
                del flc_rule.antecedents[key]
        
        # mark the table's columns that correspond with antecedents that were deleted
        for col_idx in antecedents_to_delete:
            table[:,(col_idx - 1)] = np.array([np.nan]*table.shape[0]) # offset the column index by negative 1 since we added 1 earlier
            
        # update the table to reflect the fuzzy logic controller after the deletions
        col_idx = 0
        while True:
            try:
                col = table[:,col_idx]
                if np.isnan(col).all():
                    table = np.hstack((table[:,:col_idx], table[:,(col_idx+1):]))
                else:
                    col_idx += 1
            except IndexError:
                break

        self.flc = FLC(flc_rules, table)
        return self.flc
    
    def step_5(self):
        ordered_table, ordered_rules = foo(self.flc.table, self.flc.rules)
        filtered_rules = foobar(ordered_table, ordered_rules)
        return ordered_table, filtered_rules
    
    def step_6(self, ordered_table, filtered_rules, classification=False):
        # TODO: need to add it so that all rules that have 
        # the default class as the consequent are deleted as well
        if classification:
            from common import delete_rules_with_default_consequent
            ordered_table, filtered_rules, default = delete_rules_with_default_consequent(ordered_table, filtered_rules)
        results = barfoo(ordered_table, filtered_rules)
        if classification:
            results = barbar(results, default)
        return results
    
    def to_flc(self, Z, MULTIPROCESSING=False, PROCESSES=2):
        """
        Simplifies the APFRB to a FLC.

        Parameters
        ----------
        Z : Numpy 2-D array.
            Raw data observations.
        MULTIPROCESSING : bool, optional
            Enables multiprocessing. The default is False.
        PROCESSES : int, optional
            Defines how many processes should be made if multiprocessing is enabled. The default is 2.

        Returns
        -------
        FLC
            An ordinary fuzzy logic controller.

        """

        # TODO: Exception was thrown after being called twice - replicate and fix it

        self.step_1()
        self.step_2(Z, MULTIPROCESSING, PROCESSES)
        self.step_3(Z)
        return self.step_4()
    
    def to_hflc(self, Z, classification=False):
        if self.flc is None:
            print('This RuleReducer object does not have a saved instance of a FLC. Please run \'to_flc\' first.')
        else:
            ordered_table, filtered_rules = self.step_5()
            results = self.step_6(ordered_table, filtered_rules, classification)
            
            # return results
    
            # step 7
    
            matrix = np.asmatrix(Z)
            intervals = []
            for col_idx in range(len(Z[0])):
                interval = (np.ndarray.item(min(matrix[:,col_idx])),
                            np.ndarray.item(max(matrix[:,col_idx])))
                intervals.append(interval)
            
            # TODO: TEMPORARY FIX - likely need to edit this code so it no longer has to rely on APFRB's W
            weights = deepcopy(self.apfrb.W)
    
            import sympy as sp
            from sympy.solvers import solve
            from sympy import Symbol
    
            # get the number of raw inputs
            # TODO: TEMPORARY FIX - likely need to edit this code so it no longer has to rely on APFRB's n
            n = self.apfrb.n
            argument = 'z_1:%s' % n
            # generate n number of normalized z's to use for the upcoming equation reduction
            z = sp.symbols(argument) # z is a list of size n containing all z_i variables
    
    
            # get the number of antecedents
            # TODO: TEMPORARY FIX - likely need to edit this code so it no longer has to rely on APFRB's l
            l = self.apfrb.l
            equations = []
            for j in range(l):
                equation = ''
                row_of_weights = weights[j]
                for i in range(n):
                    # z_i = z[i]
                    equation += ('Symbol("z[%s]")' % i)
                    equation += ('*(%s)' % str(row_of_weights[i]))
                    interval = intervals[i]
                    minimum = interval[0]
                    maximum = interval[1]
                    equation += ('*(%s)' % str(maximum - minimum))
                    equation += '+'
                    equation += ('(%s)' % str(minimum))
                    equation += ('*(%s)' % str(row_of_weights[i]))
                    if i + 1 < n:
                        equation += '+'
                equations.append(equation)
                    
            parsed_expressions = []
            reduced_expressions = []
            for equation_idx, equation in enumerate(equations):
                
                print('Equation %s:' % (equation_idx + 1))
                
                parsed_expressions.append(sp.sympify(equation))
    
                # get the coefficients from the equation, ignore the last coefficient in the list returned,
                # it is the constant that is not being multiplied by any normalized z_i
                coefficients = sp.Poly(equation).coeffs()
                coefficients = coefficients[:-1] # ignoring the last coefficient since it is not multiplied by any symbol
                
                # round the coefficients to avoid precision error when looking up values
                
                coefficients = [round(num, 8) for num in coefficients]
                
                threshold = 0.7 # the combined inputs have to contribute to 70% of the activation
                large_coeffs = []
                large_coeffs_indices = []
                
                copied_coefficients = deepcopy(coefficients)
                
                while True:
                    max_coeff = max(copied_coefficients, key=abs) # get the largest coefficient and keep it
                    z_idx = coefficients.index(max_coeff) + 1 # since the z_i count from 1 to n, we add plus 1
                    large_coeffs.append(max_coeff)
                    large_coeffs_indices.append(z_idx)
                    copied_coefficients.remove(max_coeff)
                    if sum(abs(np.array(large_coeffs))) / sum(abs(np.array(coefficients))) >= threshold:
                        break
                    else:
                        continue
    
                # obtain remainder of expression
                removed_part_of_expression = sp.sympify(equation)
                
                for z_idx in large_coeffs_indices:
                    arg = 'z[%s]' % (z_idx - 1)
                    # substitute the non-important weights/terms with zero
                    removed_part_of_expression = removed_part_of_expression.subs(Symbol(arg), 0)
    
                print(sp.sympify(removed_part_of_expression))
                
                print('Iterating through the data and each individual feature to remove it wherever possible...')
    
                summation = 0.0
                for observation in Z:
                    for i in range(len(observation)):
                        z_i = observation[i]
                        arg = 'z[%s]' % (i)
                        interval = intervals[i]
                        minimum = interval[0]
                        maximum = interval[1]
                        norm_z_i = (z_i - minimum) / (maximum - minimum)
                        removed_part_of_expression = removed_part_of_expression.subs(Symbol(arg), norm_z_i)
                    summation += float(removed_part_of_expression)
                summation /= len(Z)
                
                print('Creating the reduced expressions for the rule antecedents...')
    
                # create reduced expression
                reduced_expression = ''
                for idx, large_coeff in enumerate(large_coeffs):
                    z_idx = large_coeffs_indices[idx]
                    corresponding_interval_for_z = intervals[z_idx - 1]
                    minimum = corresponding_interval_for_z[0]
                    maximum = corresponding_interval_for_z[1]
                    norm_z = ('*((Symbol("z[%s]")' % (z_idx - 1))
                    norm_z += ' - %s)' % minimum
                    norm_z += (' / (%s - %s))' % (maximum, minimum))
                    reduced_expression += ('%s' % large_coeff)
                    reduced_expression += norm_z
                    # reduced_expression += ('*Symbol("z[%s]")' % (z_idx - 1))
                    if idx < len(large_coeffs) - 1:
                        reduced_expression += ' + '
                        
                reduced_expression += ('+%s' % summation)
                reduced_expressions.append(reduced_expression)
            
            # NEW CODE
            
            print('\nConverting from fuzzy logic rules to ordinary logic rules (this may take a significant amount of time)...')
            
            # go through each rule's antecedents, and substitute its antecedents with original attributes
            import re
            from rule import FLC_Rule
            from rule import OrdinaryTerm
            new_results = deepcopy(results)
            current_rule = new_results
            while True:
                try:
                    for idx, key in enumerate(current_rule.antecedents.keys()):
                        current_expression = reduced_expressions[key - 1]
                        term = current_rule.antecedents[key]
                        copied_current_expression = deepcopy(current_expression)
                        if term.type == '+':
                            copied_current_expression += ' > '
                        elif term.type == '-':
                            copied_current_expression += ' < '
                        else:
                            raise Exception('Invalid input. Something went wrong.')
                        
                        copied_current_expression += str(term.k)
                        sympy_simplified_expr = sp.simplify(sp.sympify(copied_current_expression))
                        sympy_simplified_expr = sp.factor(sp.N(sympy_simplified_expr, 8))
                        # current_rule.antecedents[key] = OrdinaryTerm(sympy_simplified_expr, key - 1) # use this for when only one large coeff is to be kept
                        
                        # current_rule.antecedents[key] = OrdinaryTerm(sympy_simplified_expr, [val - 1 for val in large_coeffs_indices]) # use this for when more than one large coeff is to be kept
                        regex = r"\[\s*\+?(-?\d+)\s*\]" # regex to extract integer from in-between square brackets
                        indices = re.findall(regex, str(list(sympy_simplified_expr.free_symbols)))
                        current_rule.antecedents[key] = OrdinaryTerm(sympy_simplified_expr, indices)

                    current_rule = current_rule.else_clause
                    
                except AttributeError:
                    if isinstance(current_rule, FLC):
                        current_rule = current_rule.rules[0]
                    else:
                        break
            return new_results
        
    def simplify(self, Z, classification=False, MULTIPROCESSING=False, PROCESSES=2):
        self.to_flc(Z, MULTIPROCESSING, PROCESSES)
        return self.to_hflc(Z, classification)