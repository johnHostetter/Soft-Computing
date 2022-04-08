import itertools

from copy import deepcopy


class Set:
    # wrapper class for Python's set, allows more complex set operations later on
    def __init__(self, elements=None, universe=None):  # universe of discourse is optional
        if elements is None:
            elements = set()
        self.X = self.elements = elements  # this must be Python set
        self.U = self.universe = universe  # note: this can sometimes be the Set class, but it should be the Python set

    def __str__(self):
        return str(self.elements)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __hash__(self):
        return hash(frozenset(self.elements))  # TODO: explore how to make this better by incorporating self.universe

    def __eq__(self, other):
        other = self.valid_binary_operation_check(other, 'Equal')
        return self.elements == other.elements

    def __sub__(self, other):
        return self.difference(other)

    def add(self, other):  # TODO: currently an unsafe operation, checks nothing about universe of discourse, etc.
        return Set(self.elements.add(other), self.universe)

    def empty(self):
        return len(self.elements) == 0

    def cardinality(self):
        return len(self.elements)

    def union(self, other):
        other = self.valid_binary_operation_check(other, 'Union')
        return Set(self.elements.union(other.elements), self.universe)

    def difference(self, other):
        other = self.valid_binary_operation_check(other, 'Difference')
        return Set(self.elements.difference(other.elements), self.universe)

    def intersection(self, other):
        other = self.valid_binary_operation_check(other, 'Intersection')
        return Set(self.elements.intersection(other.elements), self.universe)

    def cartesian_product(self, other):
        other = self.valid_binary_operation_check(other, 'Cartesian product')
        elements = set([(item_1, item_2) for item_1 in self.elements for item_2 in other.elements])
        return Set(elements, Universe(elements))

    def issubset(self, other):
        other = self.valid_binary_operation_check(other, 'Subset')
        return self.elements.issubset(other.elements)

    def valid_binary_operation_check(self, other, operation_type):
        if isinstance(other, set):  # check if the other set is a Python set, if so, update it to be Set class
            other = Set(other)
        elif isinstance(other, Set):  # check universe of discourse
            if self.universe is None and other.universe is None:
                pass  # neither object defined its universe, perform operation anyways
            elif (self.universe is None and other.universe is not None) \
                    or (self.universe is not None and other.universe is None):

                if (isinstance(self, Universe) and self.elements == other.universe.elements) \
                        or (isinstance(other, Universe) and self.universe.elements == other.elements):
                    pass  # no issues
                else:
                    # one of the sets is restricted over a universe of discourse
                    raise ValueError(
                        '%s requires the universe of discourse must be the same between sets' % operation_type)
            elif (isinstance(self, Universe) and self.elements == other.universe.elements) \
                    or (isinstance(other, Universe) and self.universe.elements == other.elements):
                pass  # no issues
            elif self.universe.elements != other.universe.elements:
                raise ValueError('%s is only defined over sets with the same universe of discourse.' % operation_type)
        else:
            raise ValueError('Instance of other object for %s is not recognized.' % operation_type)
        return other


class Universe(Set):
    # comes with special properties & operations
    def __init__(self, elements):
        super().__init__(elements)
        self.U = self.universe = self

    def __str__(self):
        return super(Set, self).__str__()

    def __truediv__(self, other):
        if isinstance(other, EquivalenceRelation):
            equivalence_classes = set()
            for element in self.elements:
                result = frozenset(other[element])
                if len(result) > 0:
                    equivalence_classes.add(result)
            return EquivalenceClassFamily(equivalence_classes, self.universe)


class Concept(Set):
    # reminder that a concept/category is a subset of universe
    def __init__(self, elements, universe):
        super().__init__(elements, universe)

        if not self.issubset(self.universe):
            raise ValueError('A set is trying to be created where its elements exceed the universe of discourse.')


class ConceptFamily:
    def __init__(self, concepts, universe):
        self.C = self.concepts = concepts
        self.U = self.universe = universe

        j = X_j = None
        coverage = Concept(set(), self.U)
        for i, X_i in enumerate(self.C):
            if issubclass(type(X_i), Set):
                X_i = Concept(X_i.elements, X_i.universe)
            if isinstance(X_i, Concept):
                # already checked that X_i is a subset of U
                coverage = coverage.union(X_i)

                if X_i.empty():  # although a concept can be empty, it cannot be empty for this
                    raise ValueError('A concept in the given concepts is not allowed to be empty.')
                elif j is not None and X_j is not None:
                    if X_i.U != X_j.U:
                        raise ValueError('The universe of discourse must be the same for concepts within the '
                                         'ConceptFamily class.')
                    if not X_i.intersection(X_j).empty():
                        raise ValueError('The intersection of two different concepts must be empty.')
            else:
                raise ValueError('A concept in the given concepts is not an instance of the Concept class.')
            j = i
            X_j = X_i

        if coverage.elements != coverage.universe:
            raise ValueError('A family of concepts must cover the entire universe of discourse.')

    def __str__(self):
        concepts_str = []
        for concept in self.concepts:
            concepts_str.append(str(concept))
        return str(concepts_str)

    def full_union(self):
        F = Concept(set(), self.universe)
        for concept in self.concepts:
            F.union(concept)
        return F

    def cardinality(self):
        return len(self.elements)

    def dispensible(self, concept):
        concepts_copy = deepcopy(self.concepts)
        concepts_copy.remove(concept)  # TODO: this currently removes the selected relation in-place
        new_family = ConceptFamily(concepts_copy, self.U)
        return self.full_union() == new_family.full_union()


class EquivalenceRelation(Concept):
    def __init__(self, relation, universe):  # the relation is a set of tuples;
        # must be reflexive, symmetric, and transitive
        self.R = relation
        # checking that the relation is a subset of the cartesian product it is defined over
        actual_universe = universe.cartesian_product(universe)
        self.R.U = self.R.universe = actual_universe
        if not self.R.issubset(actual_universe):
            raise ValueError('The equivalence relation is not a subset of its universe of discourse.')
        super().__init__(relation, actual_universe)
        self.__original_universe = universe

    def __getitem__(self, x):  # implementing [x]_{R}; where x \in U
        # (i.e., where x is an element of the universe of discourse)
        items = set()
        for pair in self.R:
            if pair[0] == x:
                for item_to_add in pair[1:]:
                    items.add(item_to_add)
        return Concept(items, self.__original_universe)


class EquivalenceClassFamily(ConceptFamily):  # an equivalence class is a concept
    def __init__(self, equivalence_class, universe):
        if isinstance(equivalence_class, ConceptFamily):
            concepts = equivalence_class.C
        else:
            concepts = []
            for category in equivalence_class:
                if isinstance(category, Concept):
                    concepts.append(category)
                elif isinstance(category, frozenset):
                    concepts.append(Concept(category, universe))
        super().__init__(concepts, universe)
        self.R = self.relation = self.make_equivalence_relation()

    def __eq__(self, other):
        if isinstance(other, EquivalenceClassFamily):
            return self.R == other.R
        else:
            raise ValueError('The EquivalenceClassFamily cannot be compared to another object that is not the same '
                             'class.')

    def make_equivalence_relation(self):
        items = set()
        equivalence_relation = set()
        for category in self.concepts:
            permutations = set(itertools.permutations(category, 2))
            equivalence_relation = equivalence_relation.union(set(permutations))
            for item in category:
                items.add(item)

        for item in items:  # add reflexive
            item_to_add = tuple([item, item])
            equivalence_relation.add(item_to_add)

        return EquivalenceRelation(Set(equivalence_relation), self.universe)


class EquivalenceRelationFamily(EquivalenceRelation):
    def __init__(self, equivalence_relations, universe):
        concepts = set()
        for x in universe:
            new_concept = None
            for equivalence_relation in equivalence_relations:
                temp_concept = equivalence_relation[x]
                if new_concept is None:
                    new_concept = temp_concept
                else:
                    new_concept = new_concept.intersection(temp_concept)
            concepts.add(new_concept)

        concept_family = ConceptFamily(concepts, universe)
        relation = EquivalenceClassFamily(concept_family, universe).make_equivalence_relation()

        super().__init__(relation, universe)
        self.equivalence_relations = equivalence_relations
        self.U = self.universe = universe

    def __getitem__(self, x):  # intersection of all equivalence relations belonging to R or "P" if P subset R
        temp = None
        for R in self.equivalence_relations:
            if temp is None:
                temp = R[x]
            else:
                temp = temp.intersection(R[x])
        return Concept(temp.elements, self.universe)

    def dispensable(self, relation):  # returns True if the given relation is dispensable, False otherwise
        IND_family = IND(self.equivalence_relations, self.equivalence_relations, self.U)
        # self.equivalence_relations.universe = relation.universe  # TODO: this isn't technically true, but it is a quick fix for now
        # IND_R = IND(self.equivalence_relations - Set(relation, relation.universe), self.equivalence_relations - Set(relation, relation.universe), self.U)

        lhs = self.U / IND_family
        relations_copy = deepcopy(self.equivalence_relations.elements)
        relations_copy.remove(relation)  # TODO: this currently removes the selected relation in-place
        new_family = EquivalenceRelationFamily(Set(relations_copy), self.U)
        IND_R = IND(new_family.equivalence_relations, new_family.equivalence_relations, self.U)
        rhs = self.U / IND_R
        return lhs == rhs

    def independent(self):  # returns True if no relation in the family is dispensable
        # aka if no relation can be removed, False otherwise
        for relation in self.equivalence_relations:
            if self.dispensable(relation):
                return False
        return True

    def RED(self):  # get all reducts of this family
        pass  # NP-hard?

    def self_intersection(self):
        result = set()
        for x in self.U:
            concept = self[x]
            result.add(concept)
        return ConceptFamily(result, self.universe)  # TODO: change this to equivalence concepts?

    def num_of_relations(self):
        return len(self.equivalence_relations)

    def elementary_categories(self):  # the family of elementary categories
        return list(itertools.combinations(self.equivalence_relations, 1))

    def basic_categories(self):  # the family of basic categories
        basic_categories = []
        for length in range(1, len(self.equivalence_relations) + 1):
            categories = list(itertools.combinations(self.equivalence_relations, length))
            basic_categories.extend(categories)
        return basic_categories


class IND(EquivalenceRelationFamily):
    def __init__(self, P, R, universe):
        if not P.empty() and P.issubset(R):
            pass
        else:
            raise ValueError("The family of equivalence relations, P, must not be empty nor exceed the subset of R.")

        self.P = P
        super().__init__(P, universe)


class KnowledgeBase:
    # Knowledge Base, K = (U, \mathbf{R})
    # s.t. U is the Universe & \mathbf{R} is a family of equivalence relations over U
    def __init__(self, universe, R):
        self.U = self.universe = universe
        self.R = self.equivalence_relation_family = R
        self.K = (self.U, self.R)

        if not (isinstance(self.U, Universe)):
            raise ValueError('The argument for the given universe must be an instance of Universe class.')
        elif self.U.empty():
            raise ValueError('The universe of discourse for the knowledge base cannot be empty.')

        if not (isinstance(self.R, EquivalenceRelationFamily)):
            raise ValueError('The argument for the given equivalence relation family must be an instance of '
                             'EquivalenceRelationFamily class.')

    def __eq__(self, other):
        if isinstance(other, KnowledgeBase):
            return IND(self.R.equivalence_relations, self.R.equivalence_relations, self.universe) == IND(
                other.R.equivalence_relations, other.R.equivalence_relations, other.universe)  # i.e., U / P = U / Q

    def Q_elementary_categories(self):  # the family of elementary categories in knowledge base K = (U, R)
        return list(itertools.combinations(self.equivalence_relation_family.equivalence_relations, 1))

    def P_basic_categories(self, P):  # the family of basic categories in knowledge base K = (U, R)
        # e.g., IND(K)
        if isinstance(P, EquivalenceRelationFamily):
            basic_categories = []
            for length in range(1, len(P) + 1):
                categories = list(
                    itertools.combinations(self.equivalence_relation_family.equivalence_relations, length))
                basic_categories.extend(categories)
            return basic_categories
        else:
            raise ValueError('P-basic categories method call requires the argument P to be an instance of the '
                             'EquivalenceRelationFamily class.')
