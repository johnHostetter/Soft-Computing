#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:30:21 2021

@author: john
"""

import time
import numpy as np

try:
    from .ann import ANN
    from .transformation import T
    from .rule_reduction import RuleReducer
    from .setup import pyrenees_classification
except ImportError:
    from ann import ANN
    from transformation import T
    from rule_reduction import RuleReducer
    from setup import pyrenees_classification
    
np.random.seed(10)

def norm_z(z, z_min, z_max):
    return (z - z_min) / (z - z_max)

def pyrenees_ann(filename):
    from tensorflow.keras.models import load_model
    model = load_model(filename)
    return ANN(model.trainable_weights[0].numpy().T, model.trainable_weights[1].numpy().T, 
              model.trainable_weights[2].numpy().T[0], model.trainable_weights[3].numpy()[0])
        
def pyrenees():
    ann = pyrenees_ann()
    apfrb = T(ann)
    print(apfrb)
    return apfrb

def eval_flc(Z, labels, data_type):
    y_pred = []
    for idx, z in enumerate(Z):
        y = []
        for j in range(ann.m):
            y.append(np.dot(ann.W[j].T, z))
        x = dict(zip(range(1, len(y) + 1), y))
        y_pred.append(flc.infer_with_u_and_d(x))
    
    transformed_y_pred = np.array([pyrenees_classification(f) for f in y_pred])
    transformed_y_student = np.array([pyrenees_classification(f) for f in ann.predict(Z)])
    
    output = []
    output.append('--- %s ---\n' % data_type)
    output.append('accuracy of FLC w.r.t. student = {}%\n'.format(100.0 * np.count_nonzero(transformed_y_pred == transformed_y_student)/len(Z)))
    output.append('accuracy of FLC w.r.t. teacher = {}%\n'.format(100.0 * np.count_nonzero(transformed_y_pred == labels)/len(Z)))
    output.append('accuracy of student w.r.t. teacher = {}%\n'.format(100.0 * np.count_nonzero(transformed_y_student == labels)/len(Z)))
    output.append('\n')
    return output

if __name__ == '__main__':  
    
    # PYRENEES EXAMPLE CODE
    
    selected_files = ['./models/seed=8_nodes=8_batch=16_patience=32_metric=mse_loss=mse.h5', 
                      './models/seed=10_nodes=8_batch=32_patience=16_metric=mse_loss=mse.h5',
                      './models/seed=20_nodes=8_batch=16_patience=32_metric=mse_loss=mse.h5',
                      './models/seed=21_nodes=8_batch=16_patience=32_metric=mse_loss=mse.h5',
                      './models/seed=22_nodes=8_batch=8_patience=32_metric=mse_loss=mse.h5',
                      './models/seed=5_nodes=6_batch=32_patience=32_metric=mse_loss=mse.h5',
                      './models/seed=8_nodes=6_batch=16_patience=16_metric=mse_loss=mse.h5',
                      './models/seed=10_nodes=6_batch=8_patience=16_metric=mse_loss=mse.h5',
                      './models/seed=16_nodes=6_batch=64_patience=32_metric=mse_loss=mse.h5',
                      './models/seed=23_nodes=6_batch=32_patience=16_metric=mse_loss=mse.h5']
    
    selected_files.reverse()
    filenames = selected_files
    
    import pandas as pd
    from common import PROBLEM_FEATURES
            
    # load training data set X and Y
    strat_train_set = pd.read_csv('./seed_2185_stratified_data/strat_train_data.csv', delimiter=',')
    strat_val_set = pd.read_csv('./seed_2185_stratified_data/strat_val_data.csv', delimiter=',')
    strat_test_set = pd.read_csv('./seed_2185_stratified_data/strat_test_data.csv', delimiter=',')
    
    X_train = strat_train_set[PROBLEM_FEATURES].values
    X_val = strat_val_set[PROBLEM_FEATURES].values
    X_test = strat_test_set[PROBLEM_FEATURES].values  
    
    y_train = strat_train_set['label'].values - 1
    y_val = strat_val_set['label'].values - 1
    y_test = strat_test_set['label'].values - 1
    
    X_critical_train = strat_train_set[strat_train_set['critical'] == 1][PROBLEM_FEATURES].values
    X_critical_val = strat_val_set[strat_val_set['critical'] == 1][PROBLEM_FEATURES].values
    X_critical_test = strat_test_set[strat_test_set['critical'] == 1][PROBLEM_FEATURES].values

    y_critical_train = strat_train_set[strat_train_set['critical'] == 1]['label'].values - 1
    y_critical_val = strat_val_set[strat_val_set['critical'] == 1]['label'].values - 1
    y_critical_test = strat_test_set[strat_test_set['critical'] == 1]['label'].values -1  
    
    for filename in filenames:
        print(filename)
        ann = pyrenees_ann(filename)
        folder_name = filename.replace('./models/', '')
        folder_name = folder_name.replace('.h5', '')
        apfrb = T(ann)
        reducer = RuleReducer(apfrb)
        
        flc = reducer.to_flc(X_train, MULTIPROCESSING=True)
        
        output = []
        output.extend(eval_flc(X_train, y_train, 'train'))
        output.extend(eval_flc(X_val, y_val, 'validation'))
        output.extend(eval_flc(X_test, y_test, 'test'))
        output.extend(eval_flc(X_critical_train, y_critical_train, 'critical train'))
        output.extend(eval_flc(X_critical_val, y_critical_val, 'critical validation'))
        output.extend(eval_flc(X_critical_test, y_critical_test, 'critical test'))

        flc.export(True, folder=folder_name)
        ann.export(folder=folder_name)
        txt_filename = r'./' + folder_name + '/README.md'
        file = open(txt_filename, 'a')
        file.writelines(output)
        file.close()
    # END OF PYRENEES EXAMPLE CODE