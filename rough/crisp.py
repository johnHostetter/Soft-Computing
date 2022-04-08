from copy import deepcopy

from essential import Set, Concept


class RestrictedSet(Concept):
    def __init__(self, elements, universe):
        super().__init__(elements, universe)


class RestrictedSetFamily:  # TODO: identical to ConceptFamily, except lose the intersection & coverage restrictions
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
                    # if not X_i.intersection(X_j).empty():
                    #     raise ValueError('The intersection of two different concepts must be empty.')
            else:
                raise ValueError('A concept in the given concepts is not an instance of the Concept class.')
            j = i
            X_j = X_i

        # if coverage.elements != coverage.universe:
        #     raise ValueError('A family of concepts must cover the entire universe of discourse.')

    def __str__(self):
        concepts_str = []
        for concept in self.concepts:
            concepts_str.append(str(concept))
        return str(concepts_str)

    def __sub__(self, other):
        concepts_copy = deepcopy(self.concepts)
        concepts_copy.remove(other)  # TODO: this currently removes the selected relation in-place
        return RestrictedSetFamily(concepts_copy, self.U)

    def full_union(self):
        F = set()
        for concept in self.concepts:
            F = F.union(concept.elements)
        return Set(F, self.universe)

    def full_intersection(self):
        F = None
        for concept in self.concepts:
            if F is None:
                F = concept.elements
            else:
                F = F.intersection(concept.elements)
        return Set(F, self.universe)

    def cardinality(self):
        return len(self.elements)

    def dispensible(self, concept):
        concepts_copy = deepcopy(self.concepts)
        concepts_copy.remove(concept)  # TODO: this currently removes the selected relation in-place
        new_family = ConceptFamily(concepts_copy, self.U)
        return self.full_union() == new_family.full_union()