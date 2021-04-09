#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:21:15 2021

@author: john
"""

import re
import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from make_train_val_test import make_train_val_test
# from neural_network import teacher
from common import PROBLEM_FEATURES
from setup import pyrenees_classification
from pyrenees_ann_to_flc import pyrenees_ann
from transformation import T
from rule_reduction import RuleReducer
from sklearn.model_selection import train_test_split


# from supervised_learning import classification, accuracy

def convert(y_est):    
    # return y_est * 0.25
    max_q_val_indices = np.argmax(y_est, axis=-1)
    new_qs = []
    for i in range(len(y_est)):
        qs = y_est[i]
        est = []
        for j in range(len(qs)):
            if j == max_q_val_indices[i]: # could perhaps encode the 'second best' q value estimate to zero
                est.append(1.0)
            else:
                est.append(-1.0)
        new_qs.append(est)
    return np.array(np.matrix(new_qs)) 
    
def normalize(train, val, test):    
    min_vector = train[PROBLEM_FEATURES].min()
    max_vector = train[PROBLEM_FEATURES].max()
    
    # apply normalization 
    train[PROBLEM_FEATURES] = ((train[PROBLEM_FEATURES] - min_vector) / (max_vector - min_vector))
    val[PROBLEM_FEATURES] = ((val[PROBLEM_FEATURES] - min_vector) / (max_vector - min_vector))
    test[PROBLEM_FEATURES] = ((test[PROBLEM_FEATURES] - min_vector) / (max_vector - min_vector))
    
    return train, val, test

def eval_flc(flc, ann, Z, labels):
    y_pred = []
    for idx, z in enumerate(Z):
        y = []
        for j in range(ann.m):
            y.append(np.dot(ann.W[j].T, z))
        x = dict(zip(range(1, len(y) + 1), y))
        y_pred.append(flc.infer_with_u_and_d(x))
    
    transformed_y_pred = np.array([pyrenees_classification(f) for f in y_pred])    
    transformed_y_student = np.array([pyrenees_classification(f) for f in ann.predict(Z)])
    flc_acc_wrt_student = 100.0 * np.count_nonzero(transformed_y_pred == transformed_y_student)/len(Z)
    flc_acc_wrt_teacher = 100.0 * np.count_nonzero(transformed_y_pred == labels)/len(Z)

    return flc_acc_wrt_teacher, flc_acc_wrt_student

def stratify(raw_data, seed):
    strat_train_set, strat_val_and_test_set = train_test_split(raw_data, train_size=0.1, 
                                                       random_state=seed, 
                                                       stratify=raw_data[['label', 'critical']])
    
    return strat_train_set
    
filenames = []
possible_val_indices = list(range(5))
possible_test_indices = list(range(5))
for possible_test_index in possible_test_indices:
    for possible_val_index in possible_val_indices:
        results = sorted(glob.glob('./models_for_cv/models_test={}_val={}/*.h5'.format(possible_test_index, possible_val_index)))
        if len(results) > 0:
            filenames.extend(results)

entries = []

for idx, filename in enumerate(filenames):
    print('{}. {}'.format(idx + 1, filename))
    regex_result = re.findall(r'\d_val=\d', filename)[0]
    test_idx = int(regex_result[0])
    val_idx = int(regex_result[-1])
        
    # load training data set X and Y
    
    train_df, val_df, test_df = make_train_val_test(test_idx, val_idx)
    train_df, val_df, test_df = normalize(train_df, val_df, test_df)
    
    X_train = train_df[PROBLEM_FEATURES].values
    X_val = val_df[PROBLEM_FEATURES].values
    X_test = test_df[PROBLEM_FEATURES].values  
    
    y_train = train_df['label'].values - 1
    y_val = val_df['label'].values - 1
    y_test = test_df['label'].values - 1
    
    X_critical_train = train_df[train_df['critical'] == 1][PROBLEM_FEATURES].values
    X_critical_val = val_df[val_df['critical'] == 1][PROBLEM_FEATURES].values
    X_critical_test = test_df[test_df['critical'] == 1][PROBLEM_FEATURES].values

    y_critical_train = train_df[train_df['critical'] == 1]['label'].values - 1
    y_critical_val = val_df[val_df['critical'] == 1]['label'].values - 1
    y_critical_test = test_df[test_df['critical'] == 1]['label'].values - 1 
        
    ann = pyrenees_ann(filename)
    folder_name = filename.replace('./models_for_cv/', './flcs_for_cv/')
    folder_name = folder_name.replace('.h5', '')
    apfrb = T(ann)
    reducer = RuleReducer(apfrb)
    
    # indices_list = list(range(len(X_train)))
    # random.shuffle(indices_list)
    # indices_list = indices_list[:1000]
    # subset_X_train = X_train.ix[indices_list]
    subset_X_train = stratify(train_df, test_idx)[PROBLEM_FEATURES].values
    flc = reducer.to_flc(subset_X_train, MULTIPROCESSING=True)

    train_acc_wrt_teacher, train_acc_wrt_student = eval_flc(flc, ann, X_train, y_train)
    val_acc_wrt_teacher, val_acc_wrt_student = eval_flc(flc, ann, X_val, y_val)
    test_acc_wrt_teacher, test_acc_wrt_student = eval_flc(flc, ann, X_test, y_test)

    crit_train_acc_wrt_teacher, crit_train_acc_wrt_student = eval_flc(flc, ann, X_critical_train, y_critical_train)
    crit_val_acc_wrt_teacher, crit_val_acc_wrt_student = eval_flc(flc, ann, X_critical_val, y_critical_val)
    crit_test_acc_wrt_teacher, crit_test_acc_wrt_student = eval_flc(flc, ann, X_critical_test, y_critical_test)

    entry = {
        'file':filename, 
        'train acc. w.r.t. teacher': train_acc_wrt_teacher, 
        'train acc. w.r.t. student': train_acc_wrt_student, 
        'val acc. w.r.t. teacher': val_acc_wrt_teacher, 
        'val acc. w.r.t. student': val_acc_wrt_student, 
        'test acc. w.r.t. teacher': test_acc_wrt_teacher, 
        'test acc. w.r.t. student': test_acc_wrt_student, 
        'crit. train acc. w.r.t. teacher': crit_train_acc_wrt_teacher, 
        'crit. train acc. w.r.t. student': crit_train_acc_wrt_student, 
        'crit. val acc. w.r.t. teacher': crit_val_acc_wrt_teacher, 
        'crit. val acc. w.r.t. student': crit_val_acc_wrt_student, 
        'crit. test acc. w.r.t. teacher': crit_test_acc_wrt_teacher,
        'crit. test acc. w.r.t. student': crit_test_acc_wrt_student,
             }

    entries.append(entry)
    
    try:
        folder = re.findall(r'.*_val=\d/', folder_name)[0]
        os.makedirs(folder)
    except FileExistsError:
        print('{} already exists...'.format(folder))

    flc.export(True, folder=folder_name, auto_reply='y')
    ann.export(folder=folder_name)

df = pd.DataFrame(entries)
df.to_csv('log.csv', index=False)