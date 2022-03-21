#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:17:28 2021

@author: john
"""


class Rule:
    def __init__(self, dictionary):
        self.dictionary = dictionary # contains the antecedents and consequents (e.g. {'A':[...], 'C':[...]})
        self.strength = 1.0 # information about how good it has been judged in its life
        self.current_contribution = 0.0 # how much the rule has contributed to current actions in the episode, <= [0, 1]
        self.past_contribution = 0.0 # how much the rule has contributed to actions in the past episode, <= [0, 1]
        