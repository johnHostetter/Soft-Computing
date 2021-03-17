#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:25 2021

@author: john
"""

import unittest

from apfrb.common import subs

class TestSubs(unittest.TestCase):
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
        result = subs(False)
        self.assertEqual(result, -1.0)

if __name__ == '__main__':
    unittest.main()
