class DecisionRule:
    def __init__(self, label, antecedents, consequents):
        self.x = label
        self.C = antecedents
        self.D = consequents

    def conditions(self):  # d_x | C
        return self.C

    def decisions(self):  # d_x | D
        return self.D


class DecisionTable:
    # special type of knowledge representation system, let K = (U, A) be a knowledge representation system
    # and let C, D proper subset of A called condition & decision attributes, respectively
    # decision table is denoted as T = (U, A, C, D)
    # equivalence classes of the relations IND(C) & IND(D) will be called condition & decision classes, respectively
    # with every x \in U we associate a function d_x: A --> V s.t. d_x(a) = a(x) for every a \in C \cup D;
    # the function d_x will be called a "decision rule" (in T),
    # and x will be referred to as a label of the decision rule d_x

    # NOTE: x, or elements in the set U in decision table do not refer to real objects
    # but are identifiers to decision rules

    # if d_x is a decision rule, then the restriction of d_x to C, denoted as d_x | C,
    # and the restriction of d_x to D, denoted d_x | D
    # will be called conditions and decisions (actions) of d_x respectively
    def __init__(self, decision_rule_labels, decision_rules, condition_attributes, decision_attributes):
        """

        Parameters
        ----------
        decision_rule_labels (U): set
        decision_rules (ds): dictionary; key is decision rule label and value is decision rule class
        condition_attributes (C): set
        decision_attributes (D): set
        """
        self.U = decision_rule_labels
        self.ds = decision_rules
        self.A = condition_attributes.union(decision_attributes)
        self.C = condition_attributes
        self.D = decision_attributes

    def consistent(self, x):  # where x is a decision rule identifier
        inconsistent = False
        d_x = self.ds[x]
        for y in self.U:
            if y != x:
                d_y = self.ds[y]
                if d_x.conditions() == d_y.conditions() and d_x.decisions() == d_y.decisions():
                    continue  # decision rule is consistent here
                else:
                    # decision rule is inconsistent
                    inconsistent = True

        if inconsistent:
            # the decision rule is inconsistent at some point, we need to collect all rules that are inconsistent
            inconsistent_rules = [d_x]
            for y in self.U:
                if y != x:
                    d_y = self.ds[y]
                    if d_x.conditions() == d_y.conditions():
                        inconsistent_rules.append(d_y)
        else:
            inconsistent_rules = []

        return inconsistent, inconsistent_rules


def test1():
    C = {'a', 'b', 'c'}
    D = {'d', 'e'}
    T = DecisionTable({}, [], C, D)
    print(T.A == {'a', 'b', 'c', 'd', 'e'})
    return T

T = test1()