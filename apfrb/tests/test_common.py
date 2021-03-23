#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:25 2021

@author: john
"""

import unittest
import numpy as np

from apfrb.common import subs, foo

class test_subs(unittest.TestCase):
    def test_subs_False(self):
        """
        Test that False becomes -1.0.

        """
        result = subs(False)
        self.assertEqual(result, -1.0)
    def test_subs_True(self):
        """
        Test that True becomes 1.0.

        """
        result = subs(True)
        self.assertEqual(result, 1.0)

class test_foo(unittest.TestCase):
    def test_foo_does_nothing_on_2_element_vector(self):
        """
        Test that a one-dimensional vector of two
        or more elements is not affected.

        """
        input = np.array([[0], [1]])
        result = foo(input)
        self.assertTrue((result == input.astype('float32')).all())

    def test_foo_does_nothing_on_3_element_vector_1(self):
        """
        Test that a one-dimensional vector of two
        or more elements is not affected.

        """
        input = np.array([[0], [1], [0]])
        result = foo(input)
        self.assertTrue((result == input.astype('float32')).all())

    def test_foo_does_nothing_on_3_element_vector_2(self):
        """
        Test that a one-dimensional vector of two
        or more elements is not affected.

        """
        input = np.array([[1], [1], [0]])
        result = foo(input)
        self.assertTrue((result == input.astype('float32')).all())

    def test_foo_does_nothing_on_3_element_vector_3(self):
        """
        Test that a one-dimensional vector of two
        or more elements is not affected.

        """
        input = np.array([[0], [0], [1]])
        result = foo(input)
        self.assertTrue((result == input.astype('float32')).all())

    def test_foo_does_something_on_3_by_2_matrix_1(self):
        """
        Test that a 3 by 2 matrix is properly adjusted.

        """
        input = np.array([[0, 1], [1, 0], [0, 0]])
        expected = np.array([[1, np.nan], [np.nan, 1], [np.nan, 0]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

    def test_foo_does_something_on_3_by_2_matrix_2(self):
        """
        Test that a 3 by 2 matrix is properly adjusted.

        """
        input = np.array([[1, 0], [1, 1], [0, 1]])
        expected = np.array([[0, np.nan], [np.nan, 0], [np.nan, 1]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

    def test_foo_does_something_on_3_by_2_matrix_3(self):
        """
        Test that a 3 by 2 matrix is properly adjusted.

        """
        input = np.array([[1, 0], [1, 1], [0, 0]])
        expected = np.array([[0, np.nan], [np.nan, 0], [np.nan, 1]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

    def test_foo_does_something_on_4_by_2_matrix(self):
        """
        Test that a 4 by 2 matrix is properly adjusted.

        """
        input = np.array([[0, 0], [0, 0], [1, 1], [1, 0]])
        expected = np.array([[np.nan, 1], [0, np.nan], [0, np.nan], [1, np.nan]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

    def test_foo_does_something_on_4_by_3_matrix(self):
        """
        Test that a 4 by 3 matrix is properly adjusted.

        """
        input = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0]])
        expected = np.array([[np.nan, 1, np.nan], [0, np.nan, 0], [0, np.nan, 0], [1, np.nan, 0]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

    def test_foo_does_something_on_4_by_3_matrix(self):
        """
        Test that a 4 by 3 matrix is properly adjusted.

        """
        input = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0]])
        expected = np.array([[np.nan, 1, np.nan], [0, np.nan, 0], [0, np.nan, 0], [1, np.nan, 0]])
        result = foo(input)
        # self.assertTrue((result == expected.astype('float32')).all())
        np.testing.assert_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
