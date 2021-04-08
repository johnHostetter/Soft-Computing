#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:25:46 2021

@author: john
"""

import os
import numpy as np
import pandas as pd

class FLC:
    def __init__(self, rules, table):
        self.rules = rules
        self.table = table
        self.d_memo = {}
        
    def __str__(self):
        """
        Get the Fuzzy Rule Base as a list of strings.

        Returns
        -------
        frb : list
            The Fuzzy Rule Base that is a list of rules formatted as strings.

        """
        return '\n'.join([str(rule) for rule in self.rules])
        
    def u(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        u = 0.0
        q = len(self.rules)
        for i in range(q):
            u += self.rules[i].t(x) * self.rules[i].consequent()
        return u

    def d(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        d : TYPE
            DESCRIPTION.

        """
        key = hash(tuple(map(float, x)))
        if key in self.d_memo:
            return self.d_memo[key]
        else:
            d = 0.0
            q = len(self.rules)
            for i in range(q):
                d += self.rules[i].t(x)
            self.d_memo[key] = d
            return d
        
    def infer_with_u_and_d(self, x):
        """
        Conducts the FLC's fuzzy inference and defuzzification when given an input 'x'.
        Capable of execution when the FLC is no longer equivalent to its previous APFRB/ANN.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.u(x) / self.d(x)
    
    def predict_with_ann(self, Z, ann, func):
        """
        Conduct fuzzy inference on each observation in the data set Z.
        
        CAUTION: This method should only be called on a Fuzzy Logic Controller AFTER it is no 
        longer equivalent to the APFRB it originated from. Otherwise, when self.infer_with_u_and_d()
        is called, it will receive a dictionary object. The code then calls the private method self.__u(), 
        and later reaches self.rules[i].t(x), but this will raise a TypeError due to attempting to
        multiply a float with a dictionary object.

        Parameters
        ----------
        Z : 2-dimensional Numpy array
            The raw observations in the data set.
        ann : ANN
            The Artificial Neural Network this Fuzzy Logic Controller was created from.
        func : function, optional
            A function that defines the mapping from float to labels.

        Returns
        -------
        1-dimensional Numpy array
            Array containing predictions for their corresponding observations in the data set Z.

        """
        predictions = []
        for z in Z:
            y = [np.dot(ann.W[j].T, z) for j in range(ann.m)]
            x = dict(zip(range(1, len(y) + 1), y))
            f = self.infer_with_u_and_d(x)
            prediction = func(f)
            predictions.append(prediction)
        return np.array(predictions)
    
    def export_terms(self, to_csv=False, folder='./flc/', filename='terms.csv'): # if to_csv is not False, provide file name
        terms = []
        n_columns = self.table.shape[1]
        for col_idx in range(n_columns):
            for term_idx in [0, 1]:
                rule_index = np.where(self.table[:,col_idx]==term_idx)[0][0] # get the first one, doesn't matter
                rule = self.rules[rule_index]
                antecedent_index = list(rule.antecedents.keys())[col_idx]
                terms.append(rule.antecedents[antecedent_index].export())
        df = pd.DataFrame(terms)
        if to_csv:
            filepath = './' + folder + '/' + filename
            df.to_csv(filepath, index=False)
            print('FLC terms successfully saved.')
        return df
    
    def export_rules(self, to_csv=False, folder='./flc/', filename='rules.csv'): # if to_csv is not False, provide file name
        rules = []
        for rule in self.rules:
            rules.append(rule.export())
        df = pd.DataFrame(rules)
        if to_csv:
            filepath = './' + folder + '/' + filename
            df.to_csv(filepath, index=False)
            print('FLC rules successfully saved.')
        return df
    
    def export(self, to_csv=False, folder='./flc/'):
        try:
            os.mkdir(folder)
        except FileExistsError:
            print('Folder already exists. Would you like to override the folder "%s"? [y/n]' % folder)
            usr_input = input()
            if usr_input.lower() != 'y':
                print('Aborting FLC export.')
                return None
        terms_df = self.export_terms(to_csv, folder)
        rules_df = self.export_rules(to_csv, folder)
        return terms_df, rules_df