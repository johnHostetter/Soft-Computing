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
new_dt = dt.remove_unnecessary_columns()
print(new_dt.data)

print(dt.decision_rule_to_family_of_sets(1))  # ['{1, 2, 3}', '{1, 4}', '{1, 2, 4, 5}']
print(dt.decision_rule_to_family_of_sets(1).full_intersection())  # == {1}
print(dt.make_core_values_matrix())  # should only produce core values for each decision rule
print(dt.decision_rule_attribute_reducts(1))  # should be [('b', 'd'), ('b', 'a')]
print(dt.simplify())  # TODO: make it so it returns the minimal decision table, as of right now, it is stochastic
