#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:32:23 2021

@author: john
"""

import numpy as np
import pandas as pd
from numpy import loadtxt

from ann import ANN
from flc import FLC
from rule import LogisticTerm, FLC_Rule

def load_flc(terms_file='./flc/terms.csv', rules_file='./flc/rules.csv'):
    terms_df = pd.read_csv(terms_file)
    rules_df = pd.read_csv(rules_file)
    rules = []
    table = []
    
    for index, row in rules_df.iterrows():
        antecedents = {}
        consequent = None
        table_entry = []
        for col_name in rules_df.columns:
            if col_name == 'consequent':
                consequent = float(row[col_name])
            else:
                term_key = row[col_name]
                term_df_entry = terms_df[terms_df['id']==term_key]
                k = term_df_entry['k'].values[0]
                neg_or_pos = term_df_entry['type'].values[0]
                if neg_or_pos == '+':
                    table_entry.append(1.0)
                else:
                    table_entry.append(0.0)
                log_term = LogisticTerm(k, neg_or_pos)
                antecedents[int(col_name)] = log_term
        rule = FLC_Rule(antecedents, consequent)
        rules.append(rule)
        table.append(table_entry)
    table = np.array(table)
    return FLC(rules, table)

def load_ann(ann_W_file='./flc/ann_W.csv', ann_b_file='./flc/ann_b.csv', 
             ann_c_file='./flc/ann_c.csv', ann_beta_file='./flc/ann_beta.csv'):
    W = loadtxt(ann_W_file, delimiter=',')
    b = loadtxt(ann_b_file, delimiter=',')
    c = loadtxt(ann_c_file, delimiter=',')
    beta = float(loadtxt(ann_beta_file, delimiter=','))
    return ANN(W, b, c, beta)

def pyrenees_classification(f):
    # with this mapping, -1 is PS, 0 is FWE, 1 is WE
    return -1 if f < -0.5 else 1 if 0.5 < f else 0

class Model:
    def __init__(self, flc, ann, func, min_vector, max_vector):
        self.flc = flc
        self.ann = ann
        self.func = func
        self.min_vector = min_vector
        self.max_vector = max_vector
        
    def predict(self, z, normalized=False):
        # z will be unnormalized, raw data observation
        # meant for true integration with Pyrenees
        if not normalized:
            norm_z = (z - self.min_vector) / (self.max_vector - self.min_vector)
        else:
            norm_z = z
        raw_pred = float(self.flc.predict_with_ann([norm_z], self.ann, self.func)[0])
        processed_pred = int(raw_pred + 1) # plus 1, since the range is {-1, 0, +1}
        return processed_pred

def load_model():
    flc = load_flc()
    ann = load_ann()
    problem_id = 'problem'
    normalization_vector_path = './pyrenees/normalization_values/normalization_features_all_{}.csv'.format(problem_id)
    df = pd.read_csv(normalization_vector_path)
    min_vector = df.min_val.values.astype(np.float64)
    max_vector = df.max_val.values.astype(np.float64)
    model = Model(flc, ann, pyrenees_classification, min_vector, max_vector)
    return model