class KnowledgeRepresentationSystem:
    # or KRS, is a pair S = (U, A)
    # where U is a nonempty, finite set called the universe
    # A is a nonempty, finite set of primitive attributes
    def __init__(self, universe, primitive_attributes):
        self.U = universe
        self.A = primitive_attributes  # each a \in self.A is a total function a: U --> V_a
        # where V_a is the set of values of a called the "domain" of a
        self.S = (self.U, self.A)
