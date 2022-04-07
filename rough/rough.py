from essential import Concept, EquivalenceRelation


class R_lower(EquivalenceRelation):
    def __init__(self, relation, X, universe):
        super().__init__(relation, universe)
        self.X = X
        elements = Concept(set(), universe)
        for x in universe:
            result = self.R[x]
            if result.issubset(X):
                elements = elements.union(result)
        self.elements = elements


class R_upper(EquivalenceRelation):
    def __init__(self, relation, X, universe):
        super().__init__(relation, universe)
        self.X = X
        elements = Concept(set(), universe)
        for x in universe:
            result = self.R[x]
            if not result.intersection(X).empty():
                elements = elements.union(result)
        self.elements = elements


class R_boundary:  # TODO: might need to inherit something?
    def __init__(self, relation, X, universe):
        self.rough_set = RoughSet(relation, X, universe)
        self.elements = self.rough_set.R_upper.elements - self.rough_set.R_lower.elements


class R_negative_region:
    def __init__(self, relation, X, universe):
        self.R_upper = R_upper(relation, X, universe)
        self.elements = universe - self.R_upper.elements


class RoughSet:
    def __init__(self, relation, X, universe):
        self.R_lower = R_lower(relation, X, universe)
        self.R_upper = R_upper(relation, X, universe)
        self.R = relation
        self.X = X
        self.universe = universe

    def accuracy_measure(self, X):  # TODO: make this so that the X is actually used
        return self.R_lower.cardinality() / self.R_upper.cardinality()

    def R_roughness_of_X(self, X):  # TODO: make this so that the X is actually used
        return 1 - self.accuracy_measure(X)