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
    from .rule import ElseRule
    from .common import bar, line
    from .transformation import APFRB_rule_to_FLC_rule
except ImportError:
    from flc import FLC
    from rule import ElseRule
    from common import bar, line
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
        table = np.matrix(self.apfrb.table) # TODO: update self.table so it is consistent with the new table
        # for i in range(len(self.table[0])):
        i = 0
        while i < len(table[0]):
            if np.all(table[:,i] == table[:,i][0]):
                print('\nDelete antecedent x_%s from all the rules.' % i)
                for flc_rule in flc_rules:
                    del flc_rule.antecedents[i + 1]
                table = np.delete(table, i, axis=1)
            else:
                i += 1

        self.flc = FLC(flc_rules, np.where(table, 1, 0))
        return self.flc

    
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
    
    def to_hflc(self, Z):
        if self.flc is None:
            print('This RuleReducer object does not have a saved instance of a FLC. Please run \'to_flc\' first.')
        else:
            # step 5
            # TODO: update this, but the below code expects 'table' to have True/False entries
            # where the FLC object stores entries in 0's or 1's (technically any integer)
            # for the time being, will use the APFRB's table since it is technically equivalent,
            # but this is risky and not very robust
            
            table = np.matrix(self.apfrb.table) # TEMPORARY FIX
            
            # table = np.matrix(self.table)
            for i in range(len(np.squeeze(np.asarray(table))[0])):
                col = np.squeeze(np.array(table[:,i]))
                uniqs, indices, counts = np.unique(col, return_index=True, return_counts=True)
                argmin = np.argmin(counts)
                argindex = indices[np.argmin(counts)]
                if min(counts) == 1:
                    least_occurring_term = uniqs[np.argmin(counts)]
                    for flc_rule in self.flc.rules:
                        # i + 1 since the count for i begins from zero, but antecedents are indexed
                        # starting from 1 in rule base. The antecedent type of a FLC rule is stored
                        # as a string, either "-" or "+", but is stored as a boolean in APFRB rule.
                        # Thus, flc_rule.antecedents[i + 1].type == "+" converts the string representation
                        # back to its boolean equivalent, and if least occurring term is True, then the
                        # term+ linguistic term is the least occurring term.
                        try:
                            key = list(flc_rule.antecedents.keys())[i] # assumes the antecedents' keys are kept "in order"
                            if flc_rule.antecedents[key].type == '+' and least_occurring_term:
                                continue # do not delete the least occurring term from the rule
                            else:
                                del flc_rule.antecedents[key]
                        except IndexError:
                            # TODO: this likely needs to be fixed, most likely the identification of
                            # antecedents in the fuzzy logic rules has some logic error that needs addressed
                            print('IndexError was thrown.')
                    # need to move the rule to the top of the rule base now (hierarchical fuzzy rule base)
                    top_flc_rule = self.flc.rules.pop(argindex)
                    top_flc_rule.else_clause = True
                    self.flc.rules.insert(0, top_flc_rule)
    
            # step 6, classification only
            consequent_frequency = {} # find the frequency for each rule's consequent term
            for flc_rule in self.flc.rules:
                try:
                    consequent_frequency[flc_rule.consequent] += 1
                except KeyError:
                    consequent_frequency[flc_rule.consequent] = 1
            # return the dictionary key that has the maximum value
            # WARNING: will only return 1 of many matches (if there is a tie), however, this is okay for this purpose
            max_freq_key = max(consequent_frequency, key=lambda k: consequent_frequency[k])
            # import operator
            # max_freq_key = max(consequent_frequency.iteritems(), key=operator.itemgetter(1))[0]
    
            index = 0
            while True:
                if index < len(self.flc.rules):
                    flc_rule = self.flc.rules[index]
                    if flc_rule.consequent == max_freq_key:
                        self.flc.rules.pop(index)
                    else:
                        index += 1
                else:
                    break
            self.flc.rules.append(ElseRule(max_freq_key))
    
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
            for equation in equations:
    #            parsed_expressions.append(sp.parse_expr(equation)) # this also works
                parsed_expressions.append(sp.sympify(equation))
    
                # get the coefficients from the equation, ignore the last coefficient in the list returned,
                # it is the constant that is not being multiplied by any normalized z_i
                coefficients = sp.Poly(equation).coeffs()
                # coefficients = sp.Poly(sp.sympify(equation)).coeffs() # this also works
                coefficients = coefficients[:-1] # ignoring the last coefficient since it is not multiplied by any symbol
                max_coeff = max(coefficients, key=abs) # get the largest coefficient and keep it
                z_idx = coefficients.index(max_coeff) + 1 # since the z_i count from 1 to n, we add plus 1
    
                # # create reduced expression
                # reduced_expression = ''
                # reduced_expression += ('%s' % max_coeff)
                # reduced_expression += ('*Symbol("z[%s]")' % (z_idx - 1))
                # reduced_expressions.append(reduced_expression)
    
                # obtain remainder of expression
                removed_part_of_expression = sp.sympify(equation)
                arg = 'z[%s]' % (z_idx - 1)
                # substitute the non-important weights/terms with zero
                removed_part_of_expression = removed_part_of_expression.subs(Symbol(arg), 0)
    
                print(sp.sympify(removed_part_of_expression))
    
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
    
                # create reduced expression
                reduced_expression = ''
                reduced_expression += ('%s' % max_coeff)
                reduced_expression += ('*Symbol("z[%s]")' % (z_idx - 1))
                reduced_expression += ('+%s' % summation)
                reduced_expressions.append(reduced_expression)
    
                # for i in range(n):
                #     if i == (z_idx - 1):
                #         # create reduced expression
                #         reduced_expression = ''
                #         reduced_expression += ('%s' % max_coeff)
                #         reduced_expression += ('*Symbol("z[%s]")' % i)
                #         reduced_expressions.append(reduced_expression)
                #     else:
                #         # obtain remainder of expression
                #         removed_part_of_expression = deepcopy(sp.parse_expr(equation))
                #         arg = 'z[%s]' % i
                #         # substitute the non-important weights/terms with zero
                #         removed_part_of_expression = removed_part_of_expression.subs(Symbol(arg), 0)
    
                # print(sp.sympify(removed_part_of_expression))
    
            return self.flc.rules, intervals, equations, reduced_expressions
        
    def simplify(self, Z, MULTIPROCESSING=False, PROCESSES=2):
        self.to_flc(Z, MULTIPROCESSING, PROCESSES)
        return self.to_hflc(Z)