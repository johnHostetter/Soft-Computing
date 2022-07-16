import pandas as pd

from decision_tables import DecisionTable
from essential import Set

df = pd.DataFrame({
    'U': [1, 2, 3, 4, 5, 6, 7],
    'a': [1, 1, 0, 1, 1, 2, 2],
    'b': [0, 0, 0, 1, 1, 1, 2],
    'c': [0, 0, 0, 0, 0, 0, 2],
    'd': [1, 0, 0, 1, 2, 2, 2],
    'e': [1, 1, 0, 0, 2, 2, 2],
})

dt = DecisionTable(df, Set({'a', 'b', 'c', 'd'}), Set({'e'}))
print(dt.data)
print(dt.make_equivalence_classes_for_attr('b'))  # ['frozenset({1, 2, 3})', 'frozenset({4, 5, 6})', 'frozenset({7})']
family = dt.make_equivalence_relation_family()
print(family)  # {(4, 4), (5, 5), (7, 7), (1, 1), (3, 3), (2, 2), (6, 6)}
print(dt.dispensable('c'))  # == True
new_dt = dt.remove_unnecessary_columns(in_place=False)
print(new_dt.data)

print(dt.decision_rule_to_family_of_sets(1))  # ['{1, 2, 3}', '{1, 4}', '{1, 2, 4, 5}']
print(dt.decision_rule_to_family_of_sets(1).full_intersection())  # == {1}
print(dt.make_core_values_matrix())  # should only produce core values for each decision rule
print(dt.decision_rule_attribute_reducts(1))  # should be [('b', 'd'), ('b', 'a')]
print(dt.simplify())  # TODO: make it so it returns the minimal decision table, as of right now, it is stochastic

# testing out the pyrenees rules

import time
import numpy as np

file_name = 'problem'
rules = pd.read_csv('{}_rules.csv'.format(file_name)).to_dict('records')

# convert the rules' antecedents and consequents references from the string representation to the Numpy format
for rule in rules:
    antecedents_indices_list = rule['A'][1:-1].split(' ')
    antecedents_indices_list = np.array(
        [int(index) for index in antecedents_indices_list])
    rule['A'] = antecedents_indices_list
    consequents_indices_list = rule['C'][1:-1].split(' ')
    consequents_indices_list = np.array(
        [int(index) for index in consequents_indices_list])
    rule['C'] = consequents_indices_list

attr_values = np.array([rule['A'] for rule in rules])
condition_attributes = Set(set(['a%s' % i for i in range(1, attr_values.shape[1])]))
decision_attributes = Set({'a130'})
df = pd.DataFrame(attr_values, columns=['a%s' % i for i in range(1, attr_values.shape[1] + 1)])
df.to_csv('rules.csv')
universe = ['x%s' % i for i in range(1, attr_values.shape[0] + 1)]
df['U'] = universe
print(df.head())
dt = DecisionTable(df[0:10], condition_attributes, decision_attributes)
start = time.time()
print(dt.simplify())
end = time.time()
print('time was %s' % (end - start))
