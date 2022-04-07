from essential import *
from crisp import *

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

print('R1 being dispensable is %s.' % equivalence_relation_family_1.dispensable(equivalence_class_1.R))
print('R is independent: %s' % equivalence_relation_family_1.independent())

# example on page 50

U = Universe(set(['x%s' % i for i in range(1, 9)]))
equivalence_class_1 = EquivalenceClassFamily({frozenset({'x1', 'x4', 'x5'}), frozenset({'x2', 'x8'}),
                                              frozenset({'x3'}), frozenset({'x6', 'x7'})}, U)
equivalence_class_2 = EquivalenceClassFamily({frozenset({'x1', 'x3', 'x5'}), frozenset({'x6'}),
                                              frozenset({'x2', 'x4', 'x7', 'x8'})}, U)
equivalence_class_3 = EquivalenceClassFamily({frozenset({'x1', 'x5'}), frozenset({'x6'}), frozenset({'x2', 'x7', 'x8'}),
                                              frozenset({'x3', 'x4'})}, U)

P = equivalence_class_1.R
Q = equivalence_class_2.R
R = equivalence_class_3.R

print('U / P = %s' % (U / P))
print('U / Q = %s' % (U / Q))
print('U / R = %s' % (U / R))

family = EquivalenceRelationFamily(Set({P, Q, R}), U)
print('U / IND({P, Q, R}) = %s' % (U / IND(family.equivalence_relations, family.equivalence_relations, U)))

print('P being dispensable is %s.' % family.dispensable(P))
print('Q being dispensable is %s.' % family.dispensable(Q))
print('R being dispensable is %s.' % family.dispensable(R))

print(EquivalenceRelationFamily(Set({P, R}), U).independent())

print(family.RED())  # TODO: implement this method

# example on page 56

X = RestrictedSet({'x1', 'x3', 'x8'}, U)
Y = RestrictedSet({'x1', 'x2', 'x4', 'x5', 'x6'}, U)
Z = RestrictedSet({'x1', 'x3', 'x4', 'x6', 'x7'}, U)
T = RestrictedSet({'x1', 'x2', 'x5', 'x7'}, U)

F = RestrictedSetFamily({X, Y, Z, T}, U)
print(F.full_union())  # should be equal to {'x1', 'x2', 'x4', 'x5', 'x7', 'x3', 'x6', 'x8'}

print((F - X).full_union())  # should be equal to {'x3', 'x5', 'x7', 'x2', 'x4', 'x1', 'x6'}
print((F - Y).full_union())  # should be equal to {'x1', 'x2', 'x4', 'x5', 'x7', 'x3', 'x6', 'x8'}
