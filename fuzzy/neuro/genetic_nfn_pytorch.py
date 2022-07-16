#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:19:15 2021

@author: john
"""

import time

from copy import deepcopy
from functools import partial

from fuzzy.neuro.adaptive import AdaptiveNeuroFuzzy
from genetic.genetic_safin import objective, genetic_algorithm

class ModifyRulesNeuroFuzzy(AdaptiveNeuroFuzzy):
    def rule_selection(self, rule_indices_to_keep):
        tmp = deepcopy(self.rules)
        kept_rules = [tmp[i] for i, index in enumerate(self.rules) if rule_indices_to_keep[i] == 1]
        self.rules = kept_rules
        tmp = deepcopy(self.weights)
        kept_weights = [tmp[i] for i, index in enumerate(self.weights) if rule_indices_to_keep[i] == 1]
        self.weights = kept_weights
        self.orphaned_term_removal()
        self.preprocessing()
        error = self.update()

        if error == -1:
            self.orphaned_term_removal()
            self.preprocessing()
            self.update()

    def rule_pruning(self, batch_X, batch_Y, batch_size, verbose=False):
        if verbose:
            print('Step 4: Pruning unnecessary fuzzy logic rules...')

        start = time.time()

        # the following commented-out code are potential ideas on fuzzy rule pruning,
        # but this function uses genetic algorithms to conduct rule pruning

        # # multiply each rules' weight by the rules' activations
        # tmp = np.nansum(1 - self.f3, axis=0) # smaller values indicate less activation
        # tmp_1 = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) # normalize the activations to [0, 1]
        # # self.weights = list(np.mean(self.f3 * np.array(self.weights), axis=0))
        # self.weights = np.array(self.weights) * np.array(tmp_1)
        # mean = np.mean(self.weights)
        # std = np.std(self.weights)
        # # evaluation = mean / std
        # # evaluation = mean # this got decent results with alpha = 0.1, beta = 0.9
        # evaluation = np.median(self.weights)

        # # choices = list(np.where(np.array(self.weights) > evaluation)[0])
        # choices = list(np.where(np.array(self.weights) <= evaluation)[0])
        # if len(choices) < batch_size:
        #     NUM_TO_DELETE = len(choices)
        # else:
        #     # NUM_TO_DELETE = batch_size
        #     NUM_TO_DELETE = len(choices)
        # rule_indices = random.sample(choices, k=int(NUM_TO_DELETE / 2))

        # define the total iterations
        n_iter = 8
        # bits
        n_bits = self.K
        # define the population size
        n_pop = 8
        # crossover rate
        r_cross = 0.9
        # mutation rate
        r_mut = 1.0 / float(n_bits)
        denominator = max(self.weights) + 0.1
        probabilities = [[(weight / denominator), 1 - (weight / denominator)] for weight in self.weights]
        best, score = genetic_algorithm(partial(objective, model=self, X=batch_X, Y=batch_Y), probabilities,
                                        n_bits, n_iter, n_pop, r_cross, r_mut)
        # print('best:')
        # print(best)
        # print('score: %s' % score)
        self.rule_selection(best)
        end = time.time()

        if verbose:
            print('%d fuzzy logic rules kept in %.2f seconds (original number is %d).' % (len(self.rules), end - start, len(best)))
