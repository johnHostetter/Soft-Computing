#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:49:48 2021

@author: john
"""

import numpy as np
import itertools
from sklearn import datasets
import matplotlib.pyplot as plt

def subs(x):
    if x:
        return 1.0
    else:
        return -1.0
    
def bar(x, heights, title, x_lbl, y_lbl):
    plt.bar(x, heights)
    plt.title(title)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()
    
def plot(x, y, title, x_lbl, y_lbl):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()
    
class Rule:
    def __init__(self, antecedents, consequents, lookup, W, v):
        self.antecedents = antecedents # dictionary
        self.consequents = consequents # dictionary
        self.lookup = lookup # lookup table for the term's linguistic meaning
        self.W = W
        self.v = v
    
    def __str__(self):
        indexes = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        
        a = list(self.consequents.values())
        signs = list(map(subs, values)) # a vector describing the +/- signs for the a's in the IF-THEN consequents
        consequent = a[0] + np.dot(signs, a[1:]) # TODO: consider storing the consequent in the rule
        output = 'IF '
        for loop_idx in range(len(values)):
            index = indexes[loop_idx]
            entry = values[loop_idx]
            if entry: # term+ is present
                output += ('x_%s is %s %.2f ' % (index, self.lookup[index - 1][1], self.v[index - 1]))
            else:
                output += ('x_%s is %s %.2f ' % (index, self.lookup[index - 1][0], self.v[index - 1]))
            if loop_idx != len(values) - 1:
                output += 'AND '
        output += 'THEN f = %.2f' % (consequent) # the consequent for the IF-THEN rule
        return output
    
    def consequent(self):
        values = list(self.antecedents.values())
        a = list(self.consequents.values())
        signs = list(map(subs, values)) # a vector describing the +/- signs for the a's in the IF-THEN consequents
        return a[0] + np.dot(signs, a[1:])
    
    def t(self, z):
        """
        Calculates the degree of firing for this rule.

        Parameters
        ----------
        z : list
            The ANN's input.

        Returns
        -------
        None.

        """
        degree = 1.0
        indexes = list(self.antecedents.keys())
        values = list(self.antecedents.values())
        for loop_idx in range(len(values)):
            index = indexes[loop_idx]
            entry = values[loop_idx]
            # y = x[index - 1]
            y = np.dot(self.W[index - 1].T, z)
            k = self.v[index - 1]
            if entry:
                degree *= self.logistic(y, k, '+')
            else:
                degree *= self.logistic(y, k)
        return degree
                
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
        self.rules = []
        if self.n > self.m:
            size = self.n
        else:
            size = self.m
        for key in range(size):
            self.lookup[key] = self.logistic_terms
        for i in range(self.r):
            rule = self.table[i] # only contains the antecedents' term assignments
            # signs = list(map(subs, rule)) # TODO: consider storing the consequent in the rule
            # consequent = self.a[0] + np.dot(signs, self.a[1:]) # TODO: consider storing the consequent in the rule
            antecedents = {(key + 1): value for key, value in enumerate(rule)} # indexed by x_i
            consequents = {key: value for key, value in enumerate(self.a)} # indexed by a_i, including a_0
            self.rules.append(Rule(antecedents, consequents, self.lookup, self.W, self.v))
            
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
            # frb.append(self.rule_str(i))
            frb.append(str(self.rules[i]))
        return '\n'.join(frb)
    
    def __c_k(self, x, k):
        diffs = []
        rule_k = self.rules[k]
        for i in range(len(self.rules)):
            rule_i = self.rules[i]
            diffs.append(abs(rule_i.consequent() - rule_k.consequent()))
        return (1/self.__d(x)) * max(diffs)
    
    def __u(self, x):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        u = 0.0
        q = self.r
        for i in range(q):
            u += self.rules[i].t(x) * self.rules[i].consequent()
        return u
    
    def __d(self, x):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        d : TYPE
            DESCRIPTION.

        """
        d = 0.0
        q = self.r
        for i in range(q):
            d += self.rules[i].t(x)
        return d
    
    def predict(self, D, func):
        predictions = []
        for z in D:
            f = self.infer_with_u_and_d(z)
            prediction = func(f)
            predictions.append(prediction)
        return np.array(predictions)
    
    def infer_with_u_and_d(self, z):
        """
        

        Parameters
        ----------
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.__u(z) / self.__d(z)
    
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
        beta = self.a[0] # fetching the output node's bias
        b = -1.0 * self.v # returning the biases to their original value
        c = self.a[1:] # fetching the weights between the hidden layer and the output node
        return ANN(self.W, b, c, beta)
    
    def simplify(self, D):
        """ step 1, for each k, if the abs(a_k) is small, 
        remove the atoms containing x_k in the IF part,
        and remove a_k from the THEN part of all the rules 
        
        step 2, for each rule k, compute m_k and l_k, 
        if m_k * l_k is small, then delete rule k from APFRB
        """
        
        # TODO: Exception was thrown after being called twice - replicate and fix it
        
        # step 1
        yes = True
        print('Would you like to remove an antecedent from the IF part? [y/n]')
        yes = input().lower() == 'y'
        while(yes):    
            print('\nStep 1 in progress (this should be quick)...')
            sorted_a = sorted([abs(x) for x in self.a[1:]]) # ignore the output node bias, find absolute values, and sort
            bar(range(len(sorted_a)), sorted_a, 'The values of a_k', 'The size of vector a (except a_0)', 'The value of a_i (where 0 < i <= m)')
            
            try:
                print('\nHow many of the smallest values would you like to retrieve? [type \'cancel\' to skip Step 1]')
                raw = input()
                ans = int(raw)
                print(sorted_a[:ans])
            except Exception:
                if raw.lower() == 'cancel':
                    print('Skipping step 1.')
                    break
                else:
                    print('Invalid response. Unsure of how to respond. Resetting step 1.')
                    continue
            
            try:
                print('\nUp to and including which value\'s index would you like to remove until? [type \'cancel\' to skip Step 1]')
                raw = input()
                ans = int(raw)
                small_val = sorted_a[ans]
                small_val_indexes = np.where(np.array(self.a[1:]) <= small_val)[0]
                # small_val_k = self.a.index(small_val)
            except Exception:
                if raw.lower() == 'cancel':
                    print('Skipping step 1.')
                    break
                else:
                    print('Invalid response. Unsure of how to respond. Resetting step 1.')
                    continue
            
            for index in small_val_indexes:
                small_val_k = self.a[index]
                for rule in self.rules:
                    try:
                        # try to delete all occurrences of x_k and a_k from any fuzzy logic rules
                        del rule.antecedents[small_val_k]
                        del rule.consequents[small_val_k]
                    except KeyError:
                        break
                    
            # iterate through the vector 'a', swapping out entries with NoneType to delete later
            for index in small_val_indexes:
                self.a[index] = None
            while True:
                try:
                    self.a.remove(None)
                except Exception:
                    break
                
            # TODO: unsure of this, but I believe that vector 'v' must also be cleaned as well
            # iterate through the vector 'v', swapping out entries with NoneType to delete later
            for index in small_val_indexes:
                self.v[index] = None
            while True:
                try:
                    self.v.remove(None)
                except Exception:
                    break
                
            print('\nWould you like to remove an antecedent from the IF part? [y/n]')
            yes = input().lower() == 'y'
            
        # step 2
        print('Step 2 in progress (this might take awhile)...')
        m_k_l_ks = []
        q = len(self.rules)
        for k in range(q):
            if k == q / 4:
                print('A quarter of the way done...')
            elif k == q / 2:
                print('Halfway done...')
            elif k == 3 * q / 4:
                print('Three quarters of the way done...')
            
            t_ks = []
            c_ks = []
            rule_k = self.rules[k]
            for z in D:
                t_ks.append(rule_k.t(z))
                c_ks.append(self.__c_k(z, k))
            m_k = max(t_ks)
            l_k = max(c_ks)
            m_k_l_ks.append(m_k * l_k)
        plt.plot(range(q), sorted(m_k_l_ks))
        plt.show() # x coordinate is the number of rules, y coordinate is m_k * l_k
        
        epsilon = 0.5 # TODO: find some way to automate this by plotting the sorted m_k * l_k's
        array = np.array(m_k_l_ks)
        indexes_to_rules_to_delete = np.where(array < epsilon)[0]
        print('There are %s fuzzy logic rules that will be deleted.' % len(indexes_to_rules_to_delete))
        # iterate through the rule base, swapping out rules with NoneType to delete later
        for rule_index in indexes_to_rules_to_delete:
            self.rules[rule_index] = None
        while True:
            try:
                self.rules.remove(None)
            except Exception:
                self.r = len(self.rules) # update the stored count of number of rules
                break
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
    b = np.array([-7, -520, -11])
    c = np.array([-0.5, 0.5, -1])
    n_inputs = 4 # number of inputs
    n_neurons = 8 # number of neurons in the hidden layer
    n_neurons = 4
    W = np.random.random(size=(n_neurons, n_inputs))
    b = np.random.random(size=(n_neurons,))
    c = np.random.random(size=(n_neurons,))
    if len(b) != len(c):
        raise Exception('The vector \'b\' must equal the vector \'c\'.')
    l = len(W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
    r = pow(2, l) # the number of fuzzy logic rules for all permutations
    return ANN(W, b, c, 0.0), l, r

def iris_classification(f):
    if f < -0.5:
        return -1 # versicolor
    elif -0.5 < f and f < 0.5:
        return 0 # virginica
    elif 0.5 < f:
        return 1 # setosa

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

# import some data to play with
iris = datasets.load_iris()
Z = iris.data[:, :4]  # we only take the first four features.
y = iris.target - 1 # target values that match APFRB paper

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
    

def test(apfrb):     
    from sklearn.neural_network import MLPClassifier
    # mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
    mlp.fit(X_train, y_train)
    # predictions = mlp.predict(X_test)
    predictions = apfrb.predict(X_test, iris_classification)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

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