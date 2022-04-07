from essential import *

# # test 1 -- meant to fail with "ValueError: The intersection of two different concepts must be empty."
# universe = {'a', 'b', 'c', 'd', 'e'}
# X_1 = Concept({'a', 'b'}, universe)
# X_2 = Concept({'b', 'c'}, {'a', 'b', 'c', 'd', 'e'})
# concepts = ConceptFamily([X_1, X_2], universe)

equivalence_class = EquivalenceClassFamily({frozenset([1, 2]), frozenset([3, 4])}, Universe({1, 2, 3, 4}))
result = equivalence_class.make_equivalence_relation()
print('result of making an equivalence relation from equivalence class = %s' % result)

# test 2 -- meant to pass
U = universe = Universe({'a', 'b', 'c'})
X_1 = Concept({'a', 'b'}, universe)
X_2 = Concept({'c'}, universe)
X_3 = Concept({'c'}, universe)
print('hashes of concepts are equal when they need to be is: %s' % (hash(X_2) == hash(X_3)))
# X_2 = Concept({'d', 'c'}, {'a', 'b', 'c', 'd', 'e'})
# X_3 = Concept({'e'}, {'a', 'b', 'c', 'd', 'e'})
concepts = ConceptFamily([X_1, X_2], universe)

R = EquivalenceRelation(Set({('a', 'a'), ('b', 'b'), ('c', 'c'), ('b', 'c'), ('c', 'b')}), universe)
print('R[a] = %s' % R['a'])
print('R[b] = %s' % R['b'])
print('R[c] = %s' % R['c'])

print('U / R = %s' % (U / R))

U = Universe(set(['x%s' % i for i in range(1, 9)]))
equivalence_class_1 = EquivalenceClassFamily({frozenset({'x1', 'x3', 'x7'}), frozenset({'x2', 'x4'}),
                                              frozenset({'x5', 'x6', 'x8'})}, U)
equivalence_class_2 = EquivalenceClassFamily({frozenset({'x1', 'x5'}), frozenset({'x2', 'x6'}),
                                              frozenset({'x3', 'x4', 'x7', 'x8'})}, U)
equivalence_class_3 = EquivalenceClassFamily({frozenset({'x2', 'x7', 'x8'}),
                                              frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})}, U)

print('U / R1 = %s' % (U / equivalence_class_1.R))
print('U / R2 = %s' % (U / equivalence_class_2.R))
print('U / R3 = %s' % (U / equivalence_class_3.R))

equivalence_relation_family_1 = EquivalenceRelationFamily(Set({equivalence_class_1.R, equivalence_class_2.R,
                                                           equivalence_class_3.R}), U)

print('1 %s' % equivalence_relation_family_1.self_intersection())

equivalence_relation_family_2 = EquivalenceRelationFamily(Set({equivalence_class_2.R, equivalence_class_3.R}), U)

print('2 %s' % equivalence_relation_family_2['x1'])

print('elementary categories = %s' % equivalence_relation_family_2.elementary_categories())
print('basic categories = %s' % equivalence_relation_family_2.basic_categories())

R = Set({equivalence_class_1.R, equivalence_class_2.R, equivalence_class_3.R})
P = Set({equivalence_class_2.R, equivalence_class_3.R})

print('IND(R): %s' % IND(R, R, U))
print('IND(P): %s' % IND(P, R, U))

print('Family of all equivalence classes of the equivalence relation IND(P): %s' % (U / IND(P, R, U)))
print('IND equivalence class for x3: %s' % IND(P, R, U)['x3'])

knowledge_base = KnowledgeBase(U, equivalence_relation_family_1)

print(knowledge_base.P_basic_categories(equivalence_relation_family_2))

# print(IND(knowledge_base)) # doesn't work yet

print(knowledge_base == knowledge_base)  # should be true
print(knowledge_base == KnowledgeBase(U, equivalence_relation_family_2))  # should be false