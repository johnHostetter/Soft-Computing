#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:28:08 2021

@author: john
"""

import unittest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

from SaFIN.clip import CLIP
from SaFIN.clip import rule_creation
from SaFIN.safin import SaFIN

class test_SaFIN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_SaFIN, self).__init__(*args, **kwargs)
        self.boston = load_boston()
        NUM_DATA = 400
        batch_X = self.boston.data[:NUM_DATA]
        batch_Y = self.boston.target[:NUM_DATA]
        ALPHA = 0.2
        BETA = 0.6
        antecedents = CLIP(batch_X, batch_Y, X_mins, X_maxes, antecedents, alpha=ALPHA, beta=BETA) # the second argument is currently not used
        consequents = CLIP(batch_Y, batch_X, Y_mins, Y_maxes, consequents, alpha=ALPHA, beta=BETA) # the second argument is currently not used

        print('--- Creating Fuzzy Logic Rules ---')

        antecedents, consequents, rules, weights = rule_creation(batch_X, batch_Y, antecedents, consequents, rules, weights)

        print('--- Number of Fuzzy Logic Rules = %d ---' % len(rules))
        consequences = [rules[idx]['C'][0] for idx in range(len(rules))]
        print(np.unique(consequences, return_counts=True))

        # make FNN

        print('--- Preprocessing Antecedents and Consequents ---')

        all_antecedents_centers = []
        all_antecedents_widths = []
        all_consequents_centers = []
        all_consequents_widths = []
        for p in range(X.shape[1]):
            antecedents_centers = [term['center'] for term in antecedents[p]]
            antecedents_widths = [term['sigma'] for term in antecedents[p]]
            all_antecedents_centers.append(antecedents_centers)
            all_antecedents_widths.append(antecedents_widths)
        for q in range(Y.shape[1]):
            consequents_centers = [term['center'] for term in consequents[q]]
            consequents_widths = [term['sigma'] for term in consequents[q]]
            all_consequents_centers.append(consequents_centers)
            all_consequents_widths.append(consequents_widths)

        term_dict = {}
        term_dict['antecedent_centers'] = boolean_indexing(all_antecedents_centers)
        term_dict['antecedent_widths'] = boolean_indexing(all_antecedents_widths)
        term_dict['consequent_centers'] = boolean_indexing(all_consequents_centers)
        term_dict['consequent_widths'] = boolean_indexing(all_consequents_widths)

        antecedents_indices_for_each_rule = np.array([rules[k]['A'] for k in range(len(rules))])
        consequents_indices_for_each_rule = np.array([rules[k]['C'] for k in range(len(rules))])

        print('--- Fuzzy Logic Rules and Linguistic Terms Ready for Neuro-Fuzzy Network ---')

        from safin import SaFIN

        fnn = SaFIN(term_dict, antecedents_indices_for_each_rule, consequents_indices_for_each_rule)

        print('--- Assembly of SaFIN Complete --- ')

        print('--- Predictions ---')

        y_predicted = fnn.feedforward(batch_X)
        rmse = np.sqrt(mean_squared_error(batch_Y, y_predicted))

        print('--- Done. RMSE = %.6f ---' % rmse)

if __name__ == '__main__':
    unittest.main()
