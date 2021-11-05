#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:28:08 2021

@author: john
"""

import numpy as np

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

from fuzzy.safin.safin import SaFIN

def main():
    boston = load_boston()
    NUM_DATA = 400
    train_X = boston.data[:NUM_DATA]
    train_Y = np.array([boston.target]).T[:NUM_DATA]
    test_X = boston.data[NUM_DATA:]
    test_Y = np.array([boston.target]).T[NUM_DATA:]
    safin = SaFIN(alpha=0.2, beta=0.6)
    _ = safin.fit(train_X, train_Y, batch_size=50, epochs=10, verbose=False, rule_pruning=False)

    # init_rmse, _ = safin.evaluate(train_X, train_Y)

    # l_rate = 0.001 # 0.1 was used for Pyrenees data
    # n_epoch = 1000
    # epsilon = 0.25
    # epoch = 0
    # curr_rmse = init_rmse
    # prev_rmse = init_rmse
    # while curr_rmse <= prev_rmse:
    #     # print('epoch %s' % epoch)
    #     y_predicted = []
    #     deltas = None
    #     for idx, x in enumerate(train_X):
    #         # print(epoch, idx)
    #         y = train_Y[idx][0]
    #         # y = Y[idx]

    #         # if idx == 59:
    #         #     print('wait')
    #         iterations = 1
    #         while True:
    #             o5 = safin.feedforward(x)
    #             consequent_delta_c, consequent_delta_widths = safin.backpropagation(x, y)
    #             if deltas is None:
    #                 deltas = {'c_c':consequent_delta_c, 'c_w':consequent_delta_widths}
    #             else:
    #                 deltas['c_c'] += consequent_delta_c
    #                 deltas['c_w'] += consequent_delta_widths
    #                 # deltas['a_c'] += antecedent_delta_c
    #                 # deltas['a_w'] += antecedent_delta_widths
    #             break
    #             # if np.abs(o5 - y) < epsilon or iterations >= 250:
    #             #     y_predicted.append(o5)
    #             #     print('achieved with %s and %s iterations' % (np.abs(o5 - y), iterations))
    #             #     break
    #             # else:
    #             #     # print(np.abs(o5 - y))
    #             #     consequent_delta_c, consequent_delta_widths, antecedent_delta_c, antecedent_delta_widths = fnn.backpropagation(x, y)
    #             #     # print(consequent_delta_c)
    #             #     # print(consequent_delta_widths)
    #             #     # print(antecedent_delta_c)
    #             #     # print(antecedent_delta_widths)

    #             #     # fnn.term_dict['consequent_centers'] += 1e-4 * consequent_delta_c
    #             #     # fnn.term_dict['consequent_widths'] += 1e-8 * consequent_delta_widths
    #             #     # fnn.term_dict['antecedent_centers'] += 1e-4 * antecedent_delta_c
    #             #     # fnn.term_dict['antecedent_widths'] += 1e-8 * antecedent_delta_widths

    #             #     fnn.term_dict['consequent_centers'] += l_rate * consequent_delta_c
    #             #     # fnn.term_dict['consequent_widths'] += 1.0 * l_rate * consequent_delta_widths
    #             #     # fnn.term_dict['antecedent_centers'] += l_rate * antecedent_delta_c
    #             #     # fnn.term_dict['antecedent_widths'] += 1.0 * l_rate * antecedent_delta_widths

    #             #     # remove anything less than or equal to zero for the linguistic term widths
    #             #     # if (fnn.term_dict['consequent_widths'] <= 0).any() or (fnn.term_dict['antecedent_widths'] <= 0).any():
    #             #     #     print('fix weights')
    #             #     # fnn.term_dict['consequent_widths'][fnn.term_dict['consequent_widths'] <= 0.0] = 1e-1
    #             #     # fnn.term_dict['antecedent_widths'][fnn.term_dict['antecedent_widths'] <= 0.0] = 1e-1

    #             #     iterations += 1
    #     safin.term_dict['consequent_centers'] -= l_rate * (deltas['c_c'] / len(train_X))
    #     safin.term_dict['consequent_widths'] -= l_rate * (deltas['c_w'] / len(train_X))
    #     # fnn.term_dict['antecedent_centers'] -= l_rate * (deltas['a_c'] / len(train_X))
    #     # fnn.term_dict['antecedent_widths'] -= l_rate * (deltas['a_w'] / len(train_X))

    #     y_predicted = []
    #     for tupl in zip(train_X, train_Y):
    #         x = tupl[0]
    #         d = tupl[1]
    #         y_predicted.append((safin.feedforward(x)))

    #     y_predicted = safin.feedforward(train_X)

    #     prev_rmse = curr_rmse
    #     curr_rmse, _ = safin.evaluate(train_X, train_Y)
    #     print('--- epoch %s --- rmse after tuning = %s (prev rmse was %s; init rmse was %s)' % (epoch, curr_rmse, prev_rmse, init_rmse))
    #     epoch += 1

    #     if curr_rmse > prev_rmse:
    #         # reverse the updates
    #         safin.term_dict['consequent_centers'] += l_rate * (deltas['c_c'] / len(train_X))
    #         safin.term_dict['consequent_widths'] += l_rate * (deltas['c_w'] / len(train_X))
    #         # fnn.term_dict['antecedent_centers'] += l_rate * (deltas['a_c'] / len(train_X))
    #         # fnn.term_dict['antecedent_widths'] += l_rate * (deltas['a_w'] / len(train_X))

    rmse = safin.evaluate(test_X, test_Y)
    print('Test RMSE = %.6f' % rmse)
    return safin

if __name__ == '__main__':
    safin = main()
