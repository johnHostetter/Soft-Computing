#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:34:13 2021

@author: john
"""

import unittest

from apfrb.apfrb import APFRB
from apfrb.ann import iris_ann, ANN
from apfrb.transformation import T, T_inv

class test_T(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_T, self).__init__(*args, **kwargs)
        self.ann = iris_ann()

    def test_ANN_to_APFRB(self):
        """
        Test that the ANN has successfully transformed to APFRB.

        """
        self.assertTrue(type(T(self.ann)) == APFRB)

class test_T_inv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_T_inv, self).__init__(*args, **kwargs)
        self.ann = iris_ann()

    def test_APFRB_to_ANN(self):
        """
        Test that the APFRB has successfully transformed to ANN.

        """
        self.assertTrue(type(T_inv(T(self.ann))) == ANN)

if __name__ == '__main__':
    unittest.main()
