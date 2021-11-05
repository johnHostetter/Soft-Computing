#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:19:15 2021

@author: john
"""

import numpy as np

from fuzzy.common.metrics import RMSE

class CoreNeuroFuzzy(object):
    """
        Contains core neuro-fuzzy network functionality.
        
        Handles fuzzification, antecedent matching, 
        fuzzy inference, aggregation and defuzzification.
        
        Note: Currently only supports limited options 
        (e.g. Singleton Fuzzifier, Gaussian membership functions, etc.).
        
        WARNING: The order in which these functions are called matters.
        They should not be called out of order, in fact, only the 'predict()'
        function call should be used.
    """
    def input_layer(self, x):
        """
        Singleton Fuzzifier (directly pass on the input vector to the next layer).

        Parameters
        ----------
        x : Numpy 2-D array
            The input vector, has a shape of (number of observations, number of inputs/attributes).

        Returns
        -------
        Numpy 2-D array
            The input vector, has a shape of (number of observations, number of inputs/attributes).

        """
        # where x is the input vector and x[i] or x_i would be the i'th element of that input vector
        # restructure the input vector into a matrix to make the condtion layer's calculations easier
        self.f1 = x
        return self.f1
    
    def condition_layer(self, o1):        
        """
        Antecedent Matching (with Gaussian membership functions).

        Parameters
        ----------
        o1 : Numpy 2-D array
            The input from the first layer, most likely the input vector(s) 
            unchanged if using Singleton Fuzzifier.

        Returns
        -------
        Numpy 2-D array
            The activations of each antecedent term in the second layer, 
            has a shape of (number of observations, number of all antecedents).

        """
        activations = np.dot(o1, self.W_1) # the shape is (num of inputs, num of all antecedents)
        
        flat_centers = self.term_dict['antecedent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['antecedent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values

        denominator = np.power(flat_widths, 2)
        denominator = np.where(denominator == 0.0, np.finfo(np.float).eps, denominator) # if there is a zero in the denominator, replace it with the smallest possible float value, otherwise, keep the other values
        self.f2 = np.exp(-1.0 * (np.power(activations - flat_centers, 2) / denominator))
            
        return self.f2 # shape is (num of inputs, num of all antecedents)
    
    def rule_base_layer(self, o2, inference='minimum'):     
        """
        Fuzzy Logic Rule Matching (with Minimum inference).

        Parameters
        ----------
        o2 : Numpy 2-D array
            The input from the second layer, most likely the activations 
            of each antecedent term in the second layer.

        Returns
        -------
        Numpy 2-D array
            The degree of applicability of each fuzzy logic rule in the third layer,
            has a shape of (number of observations, number of rules).

        """
        rule_activations = np.swapaxes(np.multiply(o2, self.W_2.T[:, np.newaxis]), 0, 1) # the shape is (num of observations, num of rules, num of antecedents)
        if inference == 'minimum':
            self.f3 = np.nanmin(rule_activations, axis=2) # the shape is (num of observations, num of rules)
        elif inference == 'product':
            self.f3 = np.nanprod(rule_activations, axis=2) # the shape is (num of observations, num of rules)
        return self.f3
    
    def consequence_layer(self, o3):   
        """
        Consequent Derivation (with Maximum T-conorm).

        Parameters
        ----------
        o3 : Numpy 2-D array
            The input from the third layer, most likely the degree of applicability
            of each fuzzy logic rule in the third layer.

        Returns
        -------
        Numpy 2-D array
            The activations of each consequent term in the fourth layer, 
            has a shape of (number of observations, number of consequent terms).

        """             
        consequent_activations = np.swapaxes(np.multiply(o3, self.W_3.T[:, np.newaxis]), 0, 1)
        self.f4 = np.nanmax(consequent_activations, axis=2)
        return self.f4
    
    def output_layer(self, o4):
        """
        Defuzzification (using Center of Averaging Defuzzifier).

        Parameters
        ----------
        o4 : Numpy 2-D array
            The input from the fourth layer, most likely the activations of each consequent
            term in the fourth layer.

        Returns
        -------
        Numpy 2-D array
            The crisp output for each output node in the fifth layer,
            has a shape of (number of observations, number of outputs).

        """
        temp_transformation = np.swapaxes(np.multiply(o4, self.W_4.T[:, np.newaxis]), 0, 1)
        
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        
        numerator = np.nansum((temp_transformation * flat_centers * flat_widths), axis=2)
        denominator = np.nansum((temp_transformation * flat_widths), axis=2)
        self.f5 = numerator / denominator
        if np.isnan(self.f5).any():
            raise Exception()
            self.f5[np.isnan(self.f5)] = 0.0 # nan values may appear if no rule in the rule base is applicable to an observation, zero out the nan values
        return self.f5

    def feedforward(self, X):
        """
        Generates output predictions for the input samples.
        
        Warning: Sensitive to the number of input samples. 
        Needs to be updated to predict using batches (i.e. may result in kernel restart).

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.

        Returns
        -------
        Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        """
        self.o1 = self.input_layer(X)
        self.o2 = self.condition_layer(self.o1)
        self.o3 = self.rule_base_layer(self.o2)
        self.o4 = self.consequence_layer(self.o3)
        self.o5 = self.output_layer(self.o4)
        return self.o5
    
    def predict(self, X):
        """
        Generates output predictions for the input samples.
        
        Warning: Sensitive to the number of input samples. 
        Needs to be updated to predict using batches (i.e. may result in kernel restart).

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.

        Returns
        -------
        Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        """
        return self.feedforward(X)
    
    def evaluate(self, X, Y):
        """
        Returns the loss value & metrics values for the model in test mode.

        Parameters
        ----------
        X : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, P), 
            where N is the number of observations, and P is the number of input features.
        Y : Numpy 2-D array.
            A Numpy 2-D array that has a shape of (N, Q),
            where N is the number of observations, and Q is the number of output features.

        Returns
        -------
        RMSE : float
            The RMSE between the target and predicted Y values.

        """
        est_Y = self.predict(X)
        return RMSE(est_Y, Y)
    
    def backpropagation(self, x, y):
        raise NotImplementedError('The backpropagation algorithm is still under development.')
        
        # (1) calculating the error signal in the output layer
        
        e5_m = y - self.o5 # y actual minus y predicted
        # e5 = np.dot(e5_m, self.W_4.T) # assign the error to its corresponding output node, shape is (num of observations, num of output nodes)
        e5 = np.multiply(e5_m[:,:,np.newaxis], self.W_4.T) # shape is (num of observations, num of output nodes, num of output terms)
        error = (self.o4 * e5.sum(axis=1))
        
        # delta centers
        flat_centers = self.term_dict['consequent_centers'].flatten()
        flat_centers = flat_centers[~np.isnan(flat_centers)] # get rid of the stored np.nan values
        flat_widths = self.term_dict['consequent_widths'].flatten()
        flat_widths = flat_widths[~np.isnan(flat_widths)] # get rid of the stored np.nan values
        # y4_k = (centers * self.W_4.T)
        # numerator = (widths * y4_k)
        widths = np.multiply(flat_widths[:,np.newaxis], self.W_4).T
        num = np.multiply(widths[np.newaxis,:,:], self.o4[:, np.newaxis,:])
        den = np.power(num.sum(axis=2), 2)
        consequent_delta_c = e5.sum(axis=1) * (num / den[:, :, np.newaxis]).sum(axis=1)
        
        # delta widths
        # c_lk = (flat_centers * self.W_4.T)
        # lhs_term = np.dot(den, c_lk)
        # rhs_term = np.multiply(num, c_lk)
        # compatible_rhs_term = rhs_term.sum(axis=1)
        # difference = lhs_term - compatible_rhs_term
        # numerator = np.multiply(self.o4, difference)
        # denominator = np.power(den, 2)
        # compatible_numerator = np.multiply(numerator[:,np.newaxis], self.W_4.T)
        # division = (compatible_numerator / denominator[:, :, np.newaxis])
        # consequent_delta_widths = division.sum(axis=1)
        
        # between consequents and outputs
        tmp = np.zeros((x.shape[0], self.Q, self.total_consequents)) # should be the same shape as self.W_4.T, but it is (num of observations, num of output nodes, num of output terms)
        # start_idx = 0
        # for q in range(self.Q):
        #     end_idx = start_idx + self.L[q]
        #     W_4[start_idx:end_idx, q] = 1
        #     start_idx = end_idx
        
        y_lk = np.swapaxes(self.o4[:,:,np.newaxis] * self.W_4, 1, 2) # shape is (num of observations, num of output nodes, num of output terms)
        c_lk = np.multiply(flat_centers[:,np.newaxis], self.W_4).T
        lhs_term = (y_lk * widths[np.newaxis,:,:])
        rhs_term = (y_lk * widths[np.newaxis,:,:] * c_lk[np.newaxis,:,:])
        for q in range(self.Q): # iterate over the output nodes
            for k in range(self.total_consequents): # iterate over their terms
                if self.W_4.T[q, k] == 1:
                    val = ((y_lk[:, q, k])[:,np.newaxis] * ((c_lk[q, k] * lhs_term.sum(axis=2)) - rhs_term.sum(axis=2)))
                    val /= np.power(lhs_term.sum(axis=2), 2)
                    tmp[:, q, k] = val[:, q]
                    
        consequent_delta_widths = e5.sum(axis=1) * tmp.sum(axis=1)
        
        return consequent_delta_c, consequent_delta_widths
