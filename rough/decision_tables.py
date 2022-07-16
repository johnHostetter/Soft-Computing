import itertools
import numpy as np
import pandas as pd

from copy import deepcopy

from crisp import RestrictedSet, RestrictedSetFamily
from essential import Set, Universe, EquivalenceClassFamily, EquivalenceRelationFamily


class DecisionTable:
    def __init__(self, pandas_dataframe, condition_attributes, decision_attributes):
        if isinstance(pandas_dataframe, pd.DataFrame):
            self.pandas_dataframe = deepcopy(pandas_dataframe)

            if 'U' in pandas_dataframe:  # must follow convention of specifying the universe of discourse
                # (i.e., each rule is an element of the universe)
                pandas_dataframe.set_index('U', inplace=True)
            else:
                raise ValueError('A column \'U\' must be specified in the given DataFrame, where each row is given a '
                                 'unique identifier in the universe of discourse.')
        else:
            raise ValueError('The argument given as a DataFrame is not an instance of the Pandas DataFrame.')

        self.data = pandas_dataframe
        self.condition_attributes = condition_attributes
        self.decision_attributes = decision_attributes
        self.all_decision_rule_ids = Universe(set(self.data.index))
        self.lookup_table = {}
        self.ordered_columns = None
        self.equivalence_relation_family = self.make_equivalence_relation_family()  # populate the lookup table

    def simplify(self, in_place=True):
        if in_place:
            # step 1: remove unnecessary columns from the decision table

            idx = 0
            num_of_remaining_columns = len(self.condition_attributes)
            prev_num_of_remaining_columns = num_of_remaining_columns + 1
            while num_of_remaining_columns < prev_num_of_remaining_columns:
            # while not self.equivalence_relation_family.independent():
                prev_num_of_remaining_columns = num_of_remaining_columns
                print('removing column %s...' % idx)
                idx += 1
                self.remove_unnecessary_columns(in_place)
                num_of_remaining_columns = len(self.condition_attributes)

            # step 2: keep only the core values

            core_values_matrix = self.make_core_values_matrix()

            # step 2a: search for the best reducts -- ideally, those that reduce the length of the rule base

            for decision_rule_id in self.all_decision_rule_ids:
                attribute_reducts = self.decision_rule_attribute_reducts(decision_rule_id)
                if len(attribute_reducts) > 0:
                    selected_reduct_index = 0  # TODO: pick the first one, it doesn't matter for now
                    selected_reduct = attribute_reducts[selected_reduct_index]

                    for condition_attribute in selected_reduct:
                        core_values_matrix.at[decision_rule_id, condition_attribute] = self.data.loc[decision_rule_id, condition_attribute]

            # step 2b: remove duplicate rules

            minimal_decision_table = core_values_matrix.drop_duplicates(
                subset=self.condition_attributes.elements)
        else:

            decision_table = self  # there is a distinct separation between self and this decision table

            # step 1: remove unnecessary columns from the decision table

            while not decision_table.equivalence_relation_family.independent():
                new_decision_table = decision_table.remove_unnecessary_columns()
                if new_decision_table is None:
                    break
                else:
                    decision_table = new_decision_table

            # step 2: keep only the core values

            core_values_matrix = decision_table.make_core_values_matrix()

            # step 2a: search for the best reducts -- ideally, those that reduce the length of the rule base

            for decision_rule_id in decision_table.all_decision_rule_ids:
                attribute_reducts = decision_table.decision_rule_attribute_reducts(decision_rule_id)
                if len(attribute_reducts) > 0:
                    selected_reduct_index = 0  # TODO: pick the first one, it doesn't matter for now
                    selected_reduct = attribute_reducts[selected_reduct_index]

                    for condition_attribute in selected_reduct:
                        core_values_matrix.at[decision_rule_id, condition_attribute] = decision_table.data.loc[decision_rule_id, condition_attribute]

            # step 2b: remove duplicate rules

            minimal_decision_table = core_values_matrix.drop_duplicates(subset=decision_table.condition_attributes.elements)

        return minimal_decision_table

    def get_decision_category(self, x):
        decision_category_relations = set()
        for decision_attribute in self.decision_attributes:
            equivalence_relation = self.lookup_table[decision_attribute]
            decision_category_relations.add(equivalence_relation)
        equivalence_relation_family = EquivalenceRelationFamily(decision_category_relations, self.all_decision_rule_ids)
        return equivalence_relation_family[x]  # return the true decision category of decision rule 'x'

    def decision_rule_indispensable_categories(self, x):
        # find the decision category that the rule cannot exceed
        true_decision_category = self.get_decision_category(x)

        indispensable_categories = set()
        # TODO: the following code makes subfamilies, similar to how reducts are found, optimize?
        for attribute_to_remove in self.condition_attributes:
            selected_attributes = self.condition_attributes - Set(attribute_to_remove)
            family_of_sets = set()
            for condition_attribute in selected_attributes:
                family_of_sets.add(RestrictedSet(self.lookup_table[condition_attribute][x].elements, self.all_decision_rule_ids))
            restricted_set_family = RestrictedSetFamily(family_of_sets, self.all_decision_rule_ids)
            decision_category = restricted_set_family.full_intersection()

            if not decision_category.issubset(true_decision_category):
                indispensable_categories.add(attribute_to_remove)

        return indispensable_categories

    def make_core_values_matrix(self):
        core_values_data = deepcopy(self.data)
        for decision_rule_id in self.all_decision_rule_ids:
            indispensable_categories = self.decision_rule_indispensable_categories(decision_rule_id)
            columns = list(self.condition_attributes - indispensable_categories)
            core_values_data.at[decision_rule_id, columns] = pd.NA
        return core_values_data

    def decision_rule_to_family_of_sets(self, x, attributes=None):
        if attributes is None:
            attributes = self.condition_attributes  # TODO: attributes can only be a subset of condition attributes
        family_of_sets = set()
        for condition_attribute in attributes:
            family_of_sets.add(RestrictedSet(self.lookup_table[condition_attribute][x].elements, self.all_decision_rule_ids))
        return RestrictedSetFamily(family_of_sets, self.all_decision_rule_ids)

    def decision_rule_attribute_reducts(self, x):
        true_decision_category = self.get_decision_category(x)
        restricted_set_family = self.decision_rule_to_family_of_sets(x)
        attribute_reducts = set()

        for L in range(1, len(self.condition_attributes) + 1):
            for subset in itertools.combinations(self.condition_attributes, L):
                restricted_set_family = self.decision_rule_to_family_of_sets(x, subset)
                decision_category = restricted_set_family.full_intersection()

                if decision_category.issubset(true_decision_category):  # this is a possible reduct
                    attribute_reducts.add(subset)

        # find the element that is the smallest w.r.t. length
        if len(attribute_reducts) > 0:
            smallest_element = min(attribute_reducts, key=len)
            # get rid of larger reducts, keep only the smallest
            attribute_reducts = [reduct for reduct in attribute_reducts if len(reduct) == len(smallest_element)]
            return attribute_reducts
        else:
            return []

    def remove_unnecessary_columns(self, in_place):
        if self.ordered_columns is None:
            # determining ordering to remove by
            map = {}
            for condition_attribute in self.condition_attributes:
                map[condition_attribute] = self.data[condition_attribute].value_counts().max() / len(self.data)

            sorted_map = sorted(map.items(), key=lambda item: item[1], reverse=True)
            self.ordered_columns = set([item[0] for item in sorted_map])

        if in_place:
            removed_attributes = set()
            for condition_attribute in self.ordered_columns:
                if self.dispensable(condition_attribute):
                    self.condition_attributes.remove(condition_attribute)  # permanently delete this attribute
                    columns_to_keep = ['U']
                    columns_to_keep.extend(self.condition_attributes)
                    columns_to_keep.extend(self.decision_attributes)

                    # update THIS decision table
                    self.pandas_dataframe = self.pandas_dataframe[columns_to_keep]
                    columns_to_keep.remove('U')
                    self.data = self.data[columns_to_keep]  # technically, shouldn't have the 'U' column included
                    del self.lookup_table[condition_attribute]
                    self.equivalence_relation_family = EquivalenceRelationFamily(Set(set(self.lookup_table.values())), self.equivalence_relation_family.universe)
                    removed_attributes.add(condition_attribute)  # add this condition attribute to remove later

            for condition_attribute in removed_attributes:
                self.ordered_columns.remove(condition_attribute)  # remove this condition attribute from the data

        else:
            for condition_attribute in self.ordered_columns:
                if self.dispensable(condition_attribute):
                    columns_to_keep = ['U']
                    columns_to_keep.extend(self.condition_attributes - Set(condition_attribute))
                    columns_to_keep.extend(self.decision_attributes)
                    self.ordered_columns.remove(condition_attribute)  # remove this condition attribute from the data
                    return DecisionTable(self.pandas_dataframe[columns_to_keep], self.condition_attributes - Set(condition_attribute), self.decision_attributes)
            return None  # None means that no new DecisionTable was made

    def dispensable(self, condition_attribute):
        R = self.lookup_table[condition_attribute]  # gets the actual equivalence relation for the given attribute
        return self.equivalence_relation_family.dispensable(R)

    def make_equivalence_classes_for_attr(self, attribute):
        classes = Set()
        for attribute_value in np.unique(self.data[attribute]):
            # decision_rule_ids \subset universe
            decision_rule_ids = frozenset(self.data.loc[self.data[attribute] == attribute_value].index)
            classes.add(decision_rule_ids)
        return EquivalenceClassFamily(classes, self.all_decision_rule_ids)

    def make_equivalence_relation_family(self):
        equivalence_relations = Set()
        for attribute in self.condition_attributes.union(self.decision_attributes):
            equivalence_class_family = self.make_equivalence_classes_for_attr(attribute)
            equivalence_relation = equivalence_class_family.R
            equivalence_relations.add(equivalence_relation)
            print(len(equivalence_relation))
            self.lookup_table[attribute] = equivalence_relation
        return EquivalenceRelationFamily(equivalence_relations, self.all_decision_rule_ids)
