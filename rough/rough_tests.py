from essential import Set, Universe, EquivalenceClassFamily
from rough import R_lower, R_upper, RoughSet
from rough import R_boundary as BN_R
from rough import R_negative_region as NEG_R


E_1 = frozenset({'x1', 'x4', 'x8'})
E_2 = frozenset({'x2', 'x5', 'x7'})
E_3 = frozenset({'x3'})
E_4 = frozenset({'x6'})
U = Universe(set(['x%s' % i for i in range(1, 9)]))

equivalence_class_1 = EquivalenceClassFamily({E_1, E_2, E_3, E_4}, U)
R = equivalence_class_1.R

print('U / R = %s' % (U / R))

X_1 = Set({'x1', 'x4', 'x7'}, U)
X_2 = Set({'x2', 'x8'}, U)
X = X_1.union(X_2)

print(X)  # result should be {'x8', 'x7', 'x2', 'x1', 'x4'}
print(R_lower(R, X, U).elements)  # result should be E_1 == {'x8', 'x1', 'x4'}
print(R_lower(R, X_1, U).elements)  # result should be set()
print(R_lower(R, X_2, U).elements)  # result should be set()

Y_1 = Set({'x1', 'x3', 'x5'}, U)
Y_2 = Set({'x2', 'x3', 'x4', 'x6'}, U)
Y = Y_1.intersection(Y_2)

print(R_upper(R, Y, U).elements)  # result should be E_3 == {'x3'}
print(R_upper(R, Y_1, U).elements)  # result should be E_1 UNION E_2 UNION E_3
print(R_upper(R, Y_2, U).elements)  # result should be (E_1 UNION E_2 UNION E_3 UNION E_4) == U

rs = RoughSet(R, X, U)
print(rs.accuracy_measure(X))

X_3 = Set({'x3', 'x6', 'x8'}, U)
print(R_lower(R, X_3, U).elements)  # result should be E_3 UNION E_4 == {'x3', 'x6'}
print(R_upper(R, X_3, U).elements)  # result should be E_1 UNION E_3 UNION E_4 == {'x1', 'x3', 'x4', 'x6', 'x8'}

print(BN_R(R, X_3, U).elements)  # result should be E_1 == {'x1', 'x4', 'x8'}
print(NEG_R(R, X_3, U).elements)  # result should be E_2 == {'x2', 'x5', 'x7'}
print(RoughSet(R, X_3, U).accuracy_measure(X_3))  # result should be 2/5

