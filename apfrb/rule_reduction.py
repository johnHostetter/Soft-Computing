#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:54:08 2021

@author: john
"""

import time
import numpy as np
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

try:
    from .common import bar, line
    from .rule import ElseRule
except ImportError:
    from common import bar, line
    from rule import ElseRule


class RuleReducer:
    def __init__(self, apfrb):
        self.apfrb = apfrb
    def step_2(self, rules, data):
        print('\nStep 2 in progress (this might take awhile)...')
        start_time = time.time()
        m_k_l_ks = []
        q = len(rules)
        for k in range(q):
            if k == q / 4:
                current_time = time.time()
                print('\nA quarter of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
            elif k == q / 2:
                current_time = time.time()
                print('\nHalfway done [elapsed time: %s seconds]...' % (current_time - start_time))
            elif k == 3 * q / 4:
                current_time = time.time()
                print('\nThree quarters of the way done [elapsed time: %s seconds]...' % (current_time - start_time))

            t_ks = []
            c_ks = []
            rule_k = rules[k]
            for z in data:
                t_ks.append(rule_k.t(z))
                c_ks.append(self.__c_k(z, k))
            m_k = max(t_ks)
            l_k = max(c_ks)
            m_k_l_ks.append(m_k * l_k)
        return m_k_l_ks
    def simplify(self, Z, MULTIPROCESSING=False):
        """ step 1, for each k, if the abs(a_k) is small,
        remove the atoms containing x_k in the IF part,
        and remove a_k from the THEN part of all the rules

        step 2, for each rule k, compute m_k and l_k,
        if m_k * l_k is small, then delete rule k from APFRB
        (WARNING: this results in a Fuzzy Logic Controller)

        step 3, if e/r is small, then output f_k(x) instead of f(x)

        step 4, if a specific atom (e.g. "x_1 is smaller than 7")
        appears in all the rules, then delete it from all of them
        """

        # TODO: Exception was thrown after being called twice - replicate and fix it

        start_time = time.time()

        # step 1
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

        # step 2
        # converts the APFRB to a FLC
        print('\nStep 2 in progress (this might take awhile)...')
        m_k_l_ks = []
        q = len(self.apfrb.rules)

        if MULTIPROCESSING:
            if __name__ == '__main__':
                with Pool(4) as p:
                    step_2 = partial(self.step_2, data=Z)
                    rules_list = [self.apfrb.rules[:int(q/4)], 
                                  self.apfrb.rules[int(q/4):int(q/2)], 
                                  self.apfrb.rules[int(q/2):3*(int(q/4))], 
                                  self.apfrb.rules[3*int(q/2):]]
                    m_k_l_ks = p.map(step_2, rules_list)
                    print(m_k_l_ks)
        else:
            for k in range(q):
                if k == q / 4:
                    current_time = time.time()
                    print('\nA quarter of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
                elif k == q / 2:
                    current_time = time.time()
                    print('\nHalfway done [elapsed time: %s seconds]...' % (current_time - start_time))
                elif k == 3 * q / 4:
                    current_time = time.time()
                    print('\nThree quarters of the way done [elapsed time: %s seconds]...' % (current_time - start_time))

                t_ks = []
                c_ks = []
                rule_k = self.apfrb.rules[k]
                for z in Z:
                    t_ks.append(rule_k.t(z))
                    c_ks.append(self.apfrb.c_k(z, k))
                m_k = max(t_ks)
                l_k = max(c_ks)
                m_k_l_ks.append(m_k * l_k)

            # x coordinate is the number of rules, y coordinate is m_k * l_k
            line(range(q), sorted(m_k_l_ks), 'The m_k * l_k of each rule', 'Rules', 'm_k * l_k')

        return m_k_l_ks
        print('\nThe five smallest m_k * l_k values: \n\n%s' % sorted(m_k_l_ks)[:5])

        epsilon = 0.3 # TODO: find some way to automate this by plotting the sorted m_k * l_k's
        array = np.array(m_k_l_ks)
        indices_to_rules_to_delete = np.where(array < epsilon)[0]
        print('\nThere are %s fuzzy logic rules that will be deleted.' % len(indices_to_rules_to_delete))
        # iterate through the rule base, swapping out rules with NoneType to delete later
        for rule_index in indices_to_rules_to_delete:
            self.rules[rule_index] = None
            self.table[rule_index] = None
        while True:
            try:
                self.rules.remove(None)
                self.table.remove(None)
            except Exception:
                self.r = len(self.rules) # update the stored count of number of rules
                break

        # step 3
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

        # beyond this point, inference no longer works
        # TODO: fix fuzzy logic inference
        flc_rules = []
        for rule in self.rules:
            flc_rules.append(rule.convert_to_flc_type())

        # step 4
        table = np.matrix(self.table) # TODO: update self.table so it is consistent with the new table
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

        # step 5
        # table = np.matrix(self.table)
        for i in range(len(np.squeeze(np.asarray(table))[0])):
            col = np.squeeze(np.array(table[:,i]))
            uniqs, indices, counts = np.unique(col, return_index=True, return_counts=True)
            argmin = np.argmin(counts)
            argindex = indices[np.argmin(counts)]
            if min(counts) == 1:
                least_occurring_term = uniqs[np.argmin(counts)]
                for flc_rule in flc_rules:
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
                top_flc_rule = flc_rules.pop(argindex)
                top_flc_rule.else_clause = True
                flc_rules.insert(0, top_flc_rule)

        # step 6, classification only
        consequent_frequency = {} # find the frequency for each rule's consequent term
        for flc_rule in flc_rules:
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
            if index < len(flc_rules):
                flc_rule = flc_rules[index]
                if flc_rule.consequent == max_freq_key:
                    flc_rules.pop(index)
                else:
                    index += 1
            else:
                break
        flc_rules.append(ElseRule(max_freq_key))

        # step 7

        matrix = np.asmatrix(Z)
        intervals = []
        for col_idx in range(len(Z[0])):
            interval = (np.ndarray.item(min(matrix[:,col_idx])),
                        np.ndarray.item(max(matrix[:,col_idx])))
            intervals.append(interval)

        weights = deepcopy(self.W)

        import sympy as sp
        from sympy.solvers import solve
        from sympy import Symbol

        # get the number of raw inputs
        n = self.n
        argument = 'z_1:%s' % n
        # generate n number of normalized z's to use for the upcoming equation reduction
        z = sp.symbols(argument) # z is a list of size n containing all z_i variables


        # get the number of antecedents
        l = self.l
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

        return flc_rules, intervals, equations, reduced_expressions