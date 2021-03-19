#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 02:09:58 2021

@author: john
"""

import unittest
import numpy as np
from sklearn import datasets

from apfrb.apfrb import APFRB
from apfrb.ann import iris_ann, ANN
from apfrb.transformation import T, T_inv
from apfrb.main import iris_labels, iris_classification

class test_ann(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_ann, self).__init__(*args, **kwargs)
        self.ann = iris_ann()
        self.apfrb = T(self.ann)
        self.iris = datasets.load_iris()
        self.Z = np.flip(self.iris.data[:, :4], axis=1)
        self.labels = np.array([iris_labels(label) for label in self.iris.target]) # target values that match APFRB paper

    def test_predict(self):
        """
        Test that the ANN's predictions are mathematically
        equivalent to the APFRB's predictions.

        """

        ann_results = np.round([self.ann.forward(z) for z in self.Z])
        apfrb_results = self.apfrb.predict(self.Z, iris_classification)
        self.assertTrue((ann_results == apfrb_results).all())

    def test_on_iris(self):
        """
        Test that the ANN's predictions solve the Iris data
        set to a satisfactory level.

        The APFRB paper reports that 99% accuracy is achieved
        with the ANN that is trained before the transformation,
        but this test uses 97% accuracy as the benchmark.

        """

        apfrb_results = self.ann.predict(self.Z, iris_classification)
        correct_predictions = np.count_nonzero(apfrb_results == self.labels)
        self.assertTrue(correct_predictions / len(self.labels) > 0.97)

if __name__ == '__main__':
    unittest.main()
