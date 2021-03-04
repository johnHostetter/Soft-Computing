#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:49:48 2021

@author: john
"""

import time
import itertools
import numpy as np
from rule import Rule, ElseRule
from common import bar, line
from copy import deepcopy
from sklearn import datasets

np.random.seed(10)     

# class FLC:
#     def __init__(self, apfrb):
#         """
#         Create a Type-1 Singleton Fuzzy Logic Controller (FLC).

#         Parameters
#         ----------
#         apfrb : APFRB
#             All Permutations Fuzzy Rule Base.

#         Returns
#         -------
#         None.

#         """
#         self.rules = self.convert_rules(apfrb.rules)
#     def convert_rules(self, rules):
#         pass

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
        self.v = list(v) # a vector of size m describing the biases for the ANN's hidden layer
        self.a = a # a vector of size m + 1 (since it includes a_0 - the output node's bias)
        self.__reset() # reset/initialize all the variables that are dependent upon 'W', 'v' or 'a'
    
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
            frb.append(str(self.rules[i]))
        return '\n'.join(frb)
    
    def __copy__(self):
        """
        Returns a shallow copy of the original APFRB.

        Returns
        -------
        newone : TYPE
            DESCRIPTION.

        """
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone
    
    def __deepcopy__(self, memo):
        """
        Returns a deep copy of the original APFRB.

        Parameters
        ----------
        memo : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        W = deepcopy(self.W)
        v = deepcopy(self.v)
        a = deepcopy(self.a)
        return APFRB(W, v, a)
    
    def __reset(self):
        """
        Resets all of the APFRB's variables excluding the matrix 'W',
        the vector 'v' and the vector 'a'. These variables are to be reset
        if 'W', 'v' or 'a' are modified at any point.

        Returns
        -------
        None.

        """
        self.linguistic_terms = 'log' # TODO: add option to make Gaussian membership functions
        self.l = len(self.W) # the number of antecedents in the fuzzy logic rules will be equal to the length of the column entries in W
        self.r = pow(2, self.l) # the number of fuzzy logic rules for all permutations
        self.m = len(self.v) # the number of neurons in the hidden layer
        self.n = len(self.W[0]) # the number of raw inputs
        self.table = list(itertools.product([False, True], repeat=self.l)) # repeat the number of times there are rule antecedents
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
            antecedents = {(key + 1): value for key, value in enumerate(rule)} # indexed by x_i
            consequents = {key: value for key, value in enumerate(self.a)} # indexed by a_i, including a_0
            self.rules.append(Rule(antecedents, consequents, self.lookup, self.W, self.v))
            
    def __delete(self, i):
        """
        Deletes the i'th entry from vector 'v' and matrix 'W', and deletes 
        the i'th + 1 entry from vector 'a' (to skip the a_0 entry aka the output's bias).
    
        CAUTION: This mutates the APFRB calling the private method. Use with extreme caution.

        Parameters
        ----------
        i : int
            i'th entry of vector 'v', matrix 'W', and i'th + 1 entry of vector 'a'.

        Returns
        -------
        None.

        """
        self.W = np.delete(self.W, i, axis=0)
        self.v = np.delete(self.v, i, axis=0)
        self.a = list(np.delete(self.a, i + 1, axis = 0))
        self.__reset()
    
    def __c_k(self, x, k):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
    
    def __b(self, x, k):
        diffs = []
        f_k = self.rules[k].t(x)
        for i in range(len(self.rules)):
            f_i = self.rules[i].t(x)
            diffs.append(abs(f_i - f_k))
        return max(diffs)
            
    def predict(self, D, func):
        predictions = []
        for z in D:
            f = self.infer_with_u_and_d(z)
            prediction = func(f)
            predictions.append(prediction)
        return np.array(predictions)
    
    def infer_with_u_and_d(self, z):
        """
        Conducts the APFRB's fuzzy inference and defuzzification when given a raw input 'z'.
        Capable of execution when the APFRB is no longer equivalent to its previous ANN.

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
        
        CAUTION: This may no longer work after simplifying the APFRB.

        Parameters
        ----------
        z : list
            Raw input provided to the ANN/APFRB.

        Raises
        ------
        Exception
            An exception is thrown when the error tolerance exceeds a constant value.

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
        try:
            b = -1.0 * self.v # returning the biases to their original value
        except TypeError: # self.v is saved as a sequence instead of a numpy array
            b = -1.0 * np.array(self.v)
        c = self.a[1:] # fetching the weights between the hidden layer and the output node
        return ANN(self.W, b, c, beta)
    
    def T_flc(self):
        pass
    
    def simplify(self, D):
        """ step 1, for each k, if the abs(a_k) is small, 
        remove the atoms containing x_k in the IF part,
        and remove a_k from the THEN part of all the rules 
        
        step 2, for each rule k, compute m_k and l_k, 
        if m_k * l_k is small, then delete rule k from APFRB
        (WARNING: this results in a Fuzzy Logic Controller)
        
        step 3, if e/r is small, then output f_k(x) instead of f(x)
        
        step 4, if a specific atom (e.g. "x_1 is smaller than 7")
        appears in all the rules, then delete it from all of them
        """
        
        # TODO: Exception was thrown after being called twice - replicate and fix it
        
        start_time = time.time()
        
        # step 1
        yes = True
        print('\nWould you like to remove an antecedent from the IF part? [y/n]')
        yes = input().lower() == 'y'
        while(yes):    
            print('\nStep 1 in progress (this should be quick)...')
            sorted_a = sorted([abs(x) for x in self.a[1:]]) # ignore the output node bias, find absolute values, and sort
            bar(range(len(sorted_a)), sorted_a, 'The values of a_k', 'The size of vector a (except a_0)', 'The value of a_i (where 0 < i <= m)')
            
            try:
                print('\nHow many of the smallest values would you like to retrieve? [type \'cancel\' to skip Step 1]')
                raw = input()
                ans = int(raw)
                print('\n%s' % sorted_a[:ans])
            except Exception:
                if raw.lower() == 'cancel':
                    print('\nSkipping step 1.')
                    break
                else:
                    print('\nInvalid response. Unsure of how to respond. Resetting step 1.')
                    continue
            
            try:
                print('\nUp to and including which value\'s index would you like to remove until? [type \'cancel\' to skip Step 1]')
                raw = input()
                ans = int(raw)
                small_val = sorted_a[ans]
                temp = np.array([abs(x) for x in self.a[1:]])
                small_val_indexes = np.where(temp <= small_val)[0]
                print('\nDeleting up to, and including, a_i = %s...' % small_val)
            except Exception:
                if raw.lower() == 'cancel':
                    print('\nSkipping step 1.')
                    break
                else:
                    print('\nInvalid response. Unsure of how to respond. Resetting step 1.')
                    continue
            
            num_of_rules_to_delete = len(self.rules) - (len(self.rules) / (2 * len(small_val_indexes)))
            print('\nConfirm the deletion of %s fuzzy logic rules (out of %s rules). [y/n]' % (num_of_rules_to_delete, len(self.rules)))
            delete = input().lower() == 'y'
            
            if delete:
                while small_val_indexes.any():
                    index = small_val_indexes[0]
                    self.__delete(index)
                    temp = np.array([abs(x) for x in self.a[1:]])
                    small_val_indexes = np.where(temp <= small_val)[0]

            #     iterate through the vector 'a', swapping out entries with NoneType to delete later
            #     TODO: unsure of this, but I believe that vector 'v' must also be cleaned as well
            #     iterate through the vector 'v', swapping out entries with NoneType to delete later
            #     self.a[index + 1] = None
            #     self.v[index] = None
                
            #     small_val_k = index + 1 # the index of x_k and a_k to be deleted
            #     for rule in self.rules:
            #         try:
            #             # try to delete all occurrences of x_k and a_k from any fuzzy logic rules
            #             del rule.antecedents[small_val_k]
            #             del rule.consequents[small_val_k]
            #         except KeyError:
            #             break
            #         finally:
            #             rule.normalize_keys()
            
            # while True:
            #     try:
            #         self.a.remove(None)
            #         self.v.remove(None)
            #     except Exception:
            #         break
        
            print('\nThe All Permutations Rule Base now has %s rules.' % len(self.rules))
            print('\nWould you like to remove another antecedent from the IF part? [y/n]')
            yes = input().lower() == 'y'
            
        # step 2
        # converts the APFRB to a FLC
        print('\nStep 2 in progress (this might take awhile)...')
        m_k_l_ks = []
        q = len(self.rules)
        for k in range(q):
            if k == q / 4:
                current_time = time.time()
                print('\nA quarter of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
            elif k == q / 2:
                current_time = time.time()
                print('\nHalfway done [elapsed time: %s seconds]...' % (current_time - start_time))
            elif k == 3 * q / 4:
                current_time = time.time()
                print('\nThree quarters of the way done [elapsed time: %s seconds]...' % (current_time - start_time))
            
            t_ks = []
            c_ks = []
            rule_k = self.rules[k]
            for z in D:
                t_ks.append(rule_k.t(z))
                c_ks.append(self.__c_k(z, k))
            m_k = max(t_ks)
            l_k = max(c_ks)
            m_k_l_ks.append(m_k * l_k)
        line(range(q), sorted(m_k_l_ks), 'The m_k * l_k of each rule', 'Rules', 'm_k * l_k') # x coordinate is the number of rules, y coordinate is m_k * l_k
        
        print('\nThe five smallest m_k * l_k values: \n\n%s' % sorted(m_k_l_ks)[:5])
        
        epsilon = 0.3 # TODO: find some way to automate this by plotting the sorted m_k * l_k's
        array = np.array(m_k_l_ks)
        indexes_to_rules_to_delete = np.where(array < epsilon)[0]
        print('\nThere are %s fuzzy logic rules that will be deleted.' % len(indexes_to_rules_to_delete))
        # iterate through the rule base, swapping out rules with NoneType to delete later
        for rule_index in indexes_to_rules_to_delete:
            self.rules[rule_index] = None
            self.table[rule_index] = None
        while True:
            try:
                self.rules.remove(None)
                self.table.remove(None)
            except Exception:
                self.r = len(self.rules) # update the stored count of number of rules
                break
        
        # step 3
        # determines whether Mean of Maximum defuzzification can be used, if e / r is small enough
        if False:
            k = None
            x = None
            e_rs = []
            
            # NOTE: any x in the training set, D, can be used. Therefore, any arbitrary x can 
            # be selected, and the for loop immediately below is not required.
            for x in D:
                xs = []
                t_ks = []
                for i in range(len(self.rules)):  
                    rule_i = self.rules[i]
                    t_ks.append(rule_i.t(x))
                    xs.append(x)
                    
                # k(x) = argmax_i t_i(x)
                k = t_ks.index(max(t_ks))
                x_ = xs[k]
                
                # compute e
                bs = []
                for x in D:
                    bs.append(self.__b(x, k))
                e = max(bs)
                
                # compute r
                vals = []
                for x in D:
                    t_k = self.rules[k].t(x)
                    denominator = 0.0
                    for i in range(len(self.rules)):
                        if i != k:
                            denominator += self.rules[i].t(x)
                    vals.append(t_k / denominator)
                r = 1.0 + min(vals)
                e_rs.append(e/r)
        
        # step 4
        table = np.matrix(self.table)
        for i in range(len(self.table[0])):
            if np.all(table[:,i] == table[:,i][0]):
                print('\nDelete antecedent x_%s from all the rules.' % i)
                # TODO: implement this step
                
        # step 5
        # beyond this point, inference no longer works
        # TODO: fix fuzzy logic inference
        flc_rules = []
        for rule in self.rules:
            flc_rules.append(rule.convert_to_flc_type())
        table = np.matrix(self.table)
        ordered_rules = []
        for i in range(len(self.table[0])):
            col = np.squeeze(np.array(table[:,i]))
            uniqs, indexes, counts = np.unique(col, return_index=True, return_counts=True)
            argmin = np.argmin(counts)
            argindex = indexes[np.argmin(counts)]
            if argmin == 1:
                least_occurring_term = uniqs[np.argmin(counts)]
                for flc_rule in flc_rules:
                    # i + 1 since the count for i begins from zero, but antecedents are indexed
                    # starting from 1 in rule base. The antecedent type of a FLC rule is stored 
                    # as a string, either "-" or "+", but is stored as a boolean in APFRB rule.
                    # Thus, flc_rule.antecedents[i + 1].type == "+" converts the string representation
                    # back to its boolean equivalent, and if least occurring term is True, then the 
                    # term+ linguistic term is the least occurring term.
                    if flc_rule.antecedents[i + 1].type == '+' and least_occurring_term:
                        continue # do not delete the least occurring term from the rule
                    else:
                        del flc_rule.antecedents[i + 1]
                # need to move the rule to the top of the rule base now (hierarchical fuzzy rule base)
                top_flc_rule = flc_rules.pop(argindex)        
                top_flc_rule.else_clause = True
                ordered_rules.insert(0, top_flc_rule)
                
        # return flc_rules
        
        # step 6, classification only
        consequent_frequency = {} # find the frequency for each rule's consequent term
        for flc_rule in flc_rules:
            try:
                consequent_frequency[flc_rule.consequent] += 1
            except KeyError:
                consequent_frequency[flc_rule.consequent] = 1
        # return the dictionary key that has the maximum value
        # WARNING: will only return 1 of many matches (if there is a tie), however, this is okay for this purpose
        max_freq_key = max(consequent_frequency, key=lambda k: consequent_frequency[k])
        # import operator
        # max_freq_key = max(consequent_frequency.iteritems(), key=operator.itemgetter(1))[0]
        
        index = 0
        while True:
            if index < len(flc_rules):
                flc_rule = flc_rules[index]
                if flc_rule.consequent == max_freq_key:
                    flc_rules.pop(index)
                else:
                    index += 1
            else:
                break
        flc_rules.append(ElseRule(max_freq_key))
        
        # step 7, 
        
        return flc_rules
                
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
    # n_inputs = 4 # number of inputs
    # n_neurons = 8 # number of neurons in the hidden layer
    # W = np.random.random(size=(n_neurons, n_inputs))
    # b = np.random.random(size=(n_neurons,))
    # c = np.random.random(size=(n_neurons,))
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
    
def avg_error(apfrb, ann, D):
    errors = []
    for x in D:
        errors.append(abs(apfrb.inference(x) - ann.forward(x)))
    return np.mean(errors)

flc_rules = apfrb.simplify(X_train)
print('\naverage error +/- %s' % avg_error(apfrb, ann, X_train))
test(apfrb)