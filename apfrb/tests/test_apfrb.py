#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 00:30:36 2021

@author: john
"""

import unittest
import numpy as np
from sklearn import datasets

from apfrb.ann import iris_ann, ANN
from apfrb.apfrb import APFRB
from apfrb.transformation import T, T_inv

class test_apfrb(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_apfrb, self).__init__(*args, **kwargs)
        self.ann = iris_ann()
        self.apfrb = T(self.ann)
        self.iris = datasets.load_iris()
        self.Z = np.flip(self.iris.data[:, :4], axis=1)
        self.labels = self.iris.target

    def test_infer_with_u_and_d(self):
        """
        Test that the APFRB's inference is mathematically equivalent to ANN's output.

        """

        ann_results = np.round([self.ann.forward(z) for z in self.Z])
        apfrb_results = np.round([self.apfrb.infer_with_u_and_d(z) for z in self.Z])
        self.assertTrue((ann_results == apfrb_results).all())

    def test_infer_with_ann(self):
        """
        Test that the APFRB's inference is mathematically equivalent to ANN's output.

        """

        ann_results = np.round([self.ann.forward(z) for z in self.Z])
        apfrb_results = np.round([self.apfrb.infer_with_ann(z) for z in self.Z])
        self.assertTrue((ann_results == apfrb_results).all())

if __name__ == '__main__':
    unittest.main()
