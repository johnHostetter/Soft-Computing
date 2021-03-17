#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:25 2021

@author: john
"""

import unittest
import numpy as np

from apfrb.rule import LogisticTerm, ElseRule
from apfrb.common import logistic

class test_LogisticTerm(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_LogisticTerm, self).__init__(*args, **kwargs)
        self.x = 0.0
        self.k = 1.0
        self.pos = '+'
        self.pos_term = LogisticTerm(self.k, self.pos)
        self.neg = "-"
        self.neg_term = LogisticTerm(self.k, self.neg)

    def test_LogisticTerm_str(self):
        """
        Test that the string of a LogisticTerm is successfully created.

        """
        self.assertEqual(str(self.pos_term), 'larger than %s' % self.k)
        self.assertEqual(str(self.neg_term), 'smaller than %s' % self.k)

    def test_LogisticTerm_mu(self):
        """
        Test that the LogisticTerm's membership function is correct.

        """
        mu = logistic(self.x, self.k, self.pos)
        self.assertEqual(self.pos_term.mu(self.x), mu)
        mu = logistic(self.x, self.k, self.neg)
        self.assertEqual(self.neg_term.mu(self.x), mu)

class test_ElseRule(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_ElseRule, self).__init__(*args, **kwargs)
        self.consequent = 'Iris flower class is Virginica'
        self.else_rule = ElseRule(self.consequent)

    def test_ElseRule_str(self):
        """
        Test that the string of an Else Rule is successfully created.
        """
        self.assertEqual(self.consequent, self.else_rule.consequent)
        self.assertEqual('ELSE %s' % self.consequent, str(self.else_rule))

if __name__ == '__main__':
    unittest.main()
