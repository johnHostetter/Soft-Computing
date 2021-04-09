# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:35:59 2021

@author: jhost
"""

import pandas as pd

def get_folds():
    k = 5
    folds = []
    for i in range(k):
        folds.append(pd.read_csv('./folds/fold{}.csv'.format(i+1)))
    return folds
    
def make_train_val_test(test_idx, val_idx):
    test_df = get_folds().pop(test_idx)
    val_df = get_folds().pop(val_idx)
    indices_to_delete = [test_idx, val_idx]
    indices_to_delete = sorted(indices_to_delete, reverse=True)
    folds = get_folds()
    print(indices_to_delete)
    for index in indices_to_delete:
        del folds[index]
    train_df = folds[0].append(folds[1]).append(folds[2])
    return train_df, val_df, test_df