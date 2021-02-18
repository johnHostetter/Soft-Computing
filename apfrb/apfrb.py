#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:49:48 2021

@author: john
"""

import numpy as np
import itertools
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
Z = iris.data[:, :4]  # we only take the first four features.
y = iris.target - 1 # target values that match APFRB paper

# # Import the model
# from sklearn.neural_network import MLPClassifier

# # Initializing the multilayer perceptron
# mlp = MLPClassifier(hidden_layer_sizes(10),solver='sgd',learning_rate_init= 0.01, max_iter=500)

# # Train the model
# mlp.fit(X_train, y_train)

# # Outputs:
# MLPClassifier (activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, 
# beta_2=0.999, early_stopping=False, epsilon=1e-08,       
# hidden_layer_sizes=10, learning_rate='constant',      
# learning_rate_init=0.01, max_iter=500, momentum=0.9,       
# nesterovs_momentum=True, power_t=0.5, random_state=None,       
# shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,       
# verbose=False, warm_start=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size = 0.20)

from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

class ANN:
    def __init__(self, W, b, c, beta):
        """
        Create an Artificial Neural Network (ANN).

        Parameters
        ----------
        W : 2-dimensional Numpy array
            The weights between the raw inputs of the ANN and the ANN's hidden layer.
        b : 1-dimensional Numpy array
            The biases for the ANN's hidden layer.
        c : 1-dimensional Numpy array
            The weights between the ANN's hidden layer and output node.
        beta : float
            The bias influencing the ANN's output node.

        Returns
        -------
        None.

        """
        self.W = W # the weights between the raw inputs and the hidden layer
        self.b = b # the biases for the hidden layer
        self.c = c # the weights between the hidden layer and the output node
        self.beta = beta # the bias for the output node
        self.m = len(self.b) # the number of neurons in the hidden layer
        self.n = len(self.W[0]) # the number of raw inputs
        
    def forward(self, z):
        """
        Conduct a forward pass in the ANN.

        Parameters
        ----------
        z : list
            Raw input provided to the ANN/APFRB.

        Returns
        -------
        f : float
            Crisp output calculated by the ANN.

        """
        f = self.beta
        y = []
        for j in range(self.m):
            y.append(np.dot(self.W[j].T, z))
            f += self.c[j] * np.tanh(y[j] + self.b[j])
        return f
    
    def T(self):
        """
        Defines the transformation between ANN to APFRB.

        Returns
        -------
        APFRB
            This ANN's equivalent APFRB.

        """
        a_0 = self.beta # assuming activation function is tanh
        a = [a_0]
        a.extend(self.c)
        v = -1.0 * self.b
        return APFRB(self.W, v, a)

class APFRB:
    def __init__(self, W, v, a):
        """
        Create an All-Permutations Fuzzy Rule Base (APFRB).

        Parameters
        ----------
        W : 2-dimensional Numpy array
            The weights between the raw inputs of the ANN and the ANN's hidden layer.
        v : 1-dimensional Numpy array
            The biases for the ANN's hidden layer.
        a : 1-dimensional Numpy array
            The weights between the ANN's hidden layer and output node.
            WARNING: first entry is the bias for the output node.

        Returns
        -------
        None.

        """
        self.W = W # the weights between the raw inputs and the hidden layer
        self.v = v # a vector of size m describing the biases for the ANN's hidden layer
        self.a = a # a vector of size m + 1 (since it includes a_0 - the output node's bias)
        self.linguistic_terms = 'log' # TODO: add option to make Gaussian membership functions
        self.l = len(self.W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
        self.r = pow(2, l) # the number of fuzzy logic rules for all permutations
        self.m = len(self.v) # the number of neurons in the hidden layer
        self.n = len(self.W[0]) # the number of raw inputs
        self.table = list(itertools.product([False, True], repeat=l)) # repeat the number of times there are rule antecedents
        self.lookup = {}
        self.logistic_terms = ['smaller than', 'larger than']
        for key in range(len(self.W[0])):
            self.lookup[key] = self.logistic_terms
            
    def rule_str(self, i):
        """
        Get the i'th fuzzy logic rule as a string type.

        Parameters
        ----------
        i : int
            Indexes the Fuzzy Rule Base and gets the i'th fuzzy logic rule.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        signs = [] # a vector describing the +/- signs for the a's in the IF-THEN consequents
        output = 'Rule %s: IF ' % (i + 1)
        rule = self.table[i]
        j = 0
        for entry in rule:
            if entry: # term+ is present
                output += ('x_%s is %s %s ' % (j + 1, self.lookup[j][1], self.v[j]))
                signs.append(1.0)
            else:
                output += ('x_%s is %s %s ' % (j + 1, self.lookup[j][0], self.v[j]))
                signs.append(-1.0)
            j += 1
            if j != len(rule):
                output += 'AND '
        output += 'THEN f = %s' % (self.a[0] + np.dot(signs, self.a[1:])) # the consequent for the IF-THEN rule
        return output
    
    def __str__(self):
        """
        Get the Fuzzy Rule Base as a list of strings.

        Returns
        -------
        frb : list
            The Fuzzy Rule Base that is a list of rules formatted as strings.

        """
        frb = []
        for i in range(self.r):
            frb.append(self.rule_str(i))
        return '\n'.join(frb)
    
    def inference(self, z):
        """
        Conducts the APFRB's fuzzy inference and defuzzification when given a raw input 'z'.

        Parameters
        ----------
        z : list
            Raw input provided to the ANN/APFRB.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        f : float
            Crisp output.

        """
        epsilon = 1e-6 # error tolerance between flc output and ann output
        f = self.a[0]
        x = []
        for j in range(self.m):
            # numerator =  0.0
            y = np.dot(self.W[j].T, z)
            x.append(y)
            t = np.tanh(x[j] - self.v[j]) # ann formula
            if True: # disable if not interested in checking FLC consistency
                # check FLC inference is still consistent with ann formula
                k = self.v[j]
                t_num = self.logistic(y, k, '+') - self.logistic(y, k)
                t_den = self.logistic(y, k, '+') + self.logistic(y, k)
                t_flc = t_num / t_den
                if abs(t_flc - t) >= epsilon:
                    raise Exception('The error tolerance of epsilon has been violated in the APFRB\'s inference.')
            j += 1 # look-ahead by 1 (to avoid the first entry which is the output node's bias)
            f += self.a[j] * t
        return f
    
    def logistic(self, y, k, t='-'):
        """
        The logistic membership function.

        Parameters
        ----------
        y : float
            The input.
        k : float
            The bias (for logistic functions, k_i = v_i for all i in 1, 2, ... m).
        t : string, optional
            Adjusts whether this logistic membership function is describing term- or term+. 
            The default is '-'.

        Returns
        -------
        float
            The degree of membership.

        """
        val = 2.0
        if t == '+':
            val = -2.0
        return 1.0 / (1.0 + np.exp(val * (y-k)))
    
    def gaussian_n(y, k):
        """
        The term- of Gaussian membership function.

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.exp(-1.0 * (pow(y - k[1], 2)) / (k[0] - k[1]))
    
    def gaussian_p(y, k):
        """
        The term+ of Gaussian membership function.

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.exp(-1.0 * (pow(y - k[0], 2)) / (k[0] - k[1]))
    
    def T_inv(self):
        """
        Defines the inverse transformation between APFRB to ANN.

        Returns
        -------
        ANN
            This APFRB's equivalent ANN.

        """
        beta = APFRB.a[0] # fetching the output node's bias
        b = -1.0 * APFRB.v # returning the biases to their original value
        c = APFRB.a[1:] # fetching the weights between the hidden layer and the output node
        return ANN(APFRB.W, b, c, beta)

def main():
    """
    The main function of this script.

    Raises
    ------
    Exception
        An exception is thrown when the vector 'b' is not equal to the vector 'c'.

    Returns
    -------
    ANN
        The ANN of interest.
    l : int
        The number of antecedents in the fuzzy logic rules.
    r : int
        The number of fuzzy logic rules for all permutations.

    """
    W = np.array([[-0.4, -5, -0.3, 0.7], [150, 150, -67, -44], [-5, 9, -7, 2]])
    # W = np.random.random(size=(8, 8))
    b = np.array([-7, -520, -11])
    c = np.array([-0.5, 0.5, -1])
    # b = np.random.random(size=(8,))
    # c = np.random.random(size=(8,))
    if len(b) != len(c):
        raise Exception('The vector \'b\' must equal the vector \'c\'.')
    l = len(W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
    r = pow(2, l) # the number of fuzzy logic rules for all permutations
    return ANN(W, b, c, 0.0), l, r

ann, l, r = main()
apfrb = ann.T()

# for x in X_train:
#     print(ann.forward(x))
#     print(apfrb.inference(x))
    
# for x in X_test:
#     print(ann.forward(x))
#     print('frb')
#     print(apfrb.inference(x))
        
def logistic_n(y, k):
    return 1 / (1 + np.exp(2 * (y-k)))
        
def logistic_p(y, k):
    return 1 / (1 + np.exp(-2 * (y-k)))

def gaussian_n(y, k):
    return np.exp(-1.0 * (pow(y - k[1], 2)) / (k[0] - k[1]))

def gaussian_p(y, k):
    return np.exp(-1.0 * (pow(y - k[0], 2)) / (k[0] - k[1]))

a_0 = 1
a_1 = 2.4

x = 1.0
k = 3.5
g = a_0 + a_1 * np.tanh(x)
f = (a_0 + a_1)*gaussian_n(x, [k, -k]) + (a_0 - a_1)*gaussian_p(x, [k, -k])
f /= gaussian_n(x, [k, -k]) + gaussian_p(x, [k, -k])

x = [1.0, 1.0]
a_0 = 1.0
a_1 = 1/3
a_2 = 2/5
x1_k = 5
x2_k = [7, 1]
g = a_0 + np.tanh(x[0] - 5)/3 + (2*(np.tanh(x[1] - 4)/5))
input1 = (logistic_p(x[0], 5) - logistic_n(x[0], 5)) / (logistic_p(x[0], 5) + logistic_n(x[0], 5))
input2 = (gaussian_p(x[1], x2_k) - gaussian_n(x[1], x2_k)) / (gaussian_p(x[1], x2_k) + gaussian_n(x[1], x2_k))
f = a_0 + a_1 * input1 + a_2 * input2