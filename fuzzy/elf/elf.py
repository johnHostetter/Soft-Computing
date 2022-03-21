#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:13:10 2021

@author: john
"""


class ELF:
    def __init__(self):
        pass
    
    def match(self, state):
        # if there are no rules matching the current world description,
        # ELF applies a "cover detector" operator, and generates a rule whose
        # antecedent matches the current state with the highest degree, in
        # accordance with constraints possibly defined by the designer
        
        # ELF selects randomly the consequents of the so-generated rules
        # the cover detector is the only operator that introduces new sub-populations;
        # it generates only the antecedents matching some state that has occurred during the learning session
        
        # the cover detector operator may also introduce, with a given probability, 
        # "don't care" symbols in the antecedents
        
        # in this case, the new rule belongs virtually to all the sub-populations compatible with the antecedent
        
        # new rules may be added to a sub-population where there are "too few" rules in a sub-population
        # with respect to the optimal number of rules for a sub-population
        
        # ELF computes this sub-optimal number of rules as a heuristic function of the 
        # present sub-population performance, and of the maximum number of rules given by the user
        
        pass
    
    def distribute(self, reinforcement):
        # after reinforcement distribution, ELF mutates with a certain probability some of the 
        # consequents of the rules that have been "tested enough", and that have a low fitness
        
        # this means that we give to a rule the possibility of demonstrating its fitness
        
        # after having contributed to enough actions, if its fitness tests out low we substitute
        # it with a rule proposing a different action
        
        # we need to try a rule some times to average possible noise
        
        # given both the small dimension of the sub-populations, and the small number of symbols
        # in the rule consequents, in almost all experiments we have decided to use only a mutation
        # operator, and not to consider the crossover operator, typical of GA and LCS
        
        # if a sub-population has "too many" rules, ELF deletes the worst of them from the rule base
        
        # the last population update mechanism fires when the population does not change for 
        # a user-defined number of episodes, and the performance is higher than a user-defined value
        
        pass
    
    def execute_action(self, state):
        # ELF randomly selects (step 3.2) one rule for each sub-population that matches the current state (step 3.1)
        # having selected the triggering rules, ELF produces the control action by combining the proposed actions
        # and defuzzyfying them (step 3.3)
        sub_populations = self.match(state) # step 3.1
        triggered_rules = self.select_triggering_rules(sub_populations) # step 3.2
        self.trigger(triggered_rules) # step 3.3
        
    def save_and_mutate(self, current_rulebase):
        if self.steady(self.current_rulebase) and self.good(self.current_rulebase): # step 7.1
            self.save(self.current_rulebase) # step 7.2
            self.mutate(self.worst_rule(self.current_rulebase)) # step 7.3
    
    def train(self):
        while not end_of_trial: # step 1
            while not end_of_episode: # step 2
                self.execute_action() # step 3
            state = self.detect_environment() # step 4
            reinforcement = self.evaluate(state) # step 5, delayed reinforcement
            self.distribute(reinforcement) # step 6
            self.save_and_mutate(self.current_rulebase) # step 7
        self.final_rulebase = self.refine(self.select_best_rulebase()) # step 8