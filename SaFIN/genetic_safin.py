#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:29:59 2021

@author: john
"""

import time
import numpy as np

# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
from copy import deepcopy

# objective function
def objective(rule_indices, model, X, Y):
    end = time.time()
    tmp = deepcopy(model)
    tmp.rule_selection(rule_indices)
    rmse, adj_rmse = tmp.evaluate(X, Y)
    cfs = [rule['CF'] for rule in tmp.rules]
    # return rmse * tmp.K * (-1*(np.sum(tmp.weights) + np.sum(cfs)))
    # t = end - np.array([rule['time_added'] for rule in tmp.rules]).min()
    # time = np.min(times)
    DESIRED_RULES = 32
    # W1 = 1e2 # this worked
    W1 = 1e3
    WEIGHT = 1e1
    print('rmse %s' % rmse)
    print('adj. rmse %s' % adj_rmse)
    print('rules %s' % (WEIGHT * (tmp.K / DESIRED_RULES)))
    print('cfs %s' % (1 - np.mean(cfs)))
    if adj_rmse is None:
        return rmse
    else:
        if adj_rmse == 0:
            print('wait')
            return rmse
        return adj_rmse
        return (W1 * adj_rmse) * (WEIGHT * (tmp.K / DESIRED_RULES)) * (1/(np.sum(tmp.weights))) * (1 - np.min(cfs))
    #     return adj_rmse * (1 - np.mean(cfs))
    # return (W1 * adj_rmse) * (WEIGHT * (tmp.K / DESIRED_RULES)) * (1/(np.sum(tmp.weights))) * (1 - np.min(cfs))

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, probabilities, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    # population = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]   
    # weighted population
    population = []
    for _ in range(n_pop):
        population.append([np.random.choice(np.arange(0, 2), p=probabilities[i]) for i in range(n_bits)])
    # keep track of best solution
    best, best_eval = 0, objective(population[0])
    best = population[best] # get the 0'th candidate if there is no best result from generations
    
    # enumerate generations
    for generation in range(n_iter):
        print(generation)
        # evaluate all candidates in the population
        scores = [objective(candidate) for candidate in population]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print("&gt;%d, new best f(%s) = %.3f" % (generation,  population[i], scores[i]))
        # select parents
        selected = [selection(population, scores) for _ in range(n_pop)]
        # create the next generation
        children= list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for candidate in crossover(p1, p2, r_cross):
                # mutation
                mutation(candidate, r_mut)
                # store for next generation
                children.append(candidate)
        # replace population
        population = children
    return [best, best_eval]