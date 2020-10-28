"""
The following code was written by Seyed Saeid Masoumzadeh (GitHub user ID: seyedsaeidmasoumzadeh),
and was published for public use on GitHub under the MIT License at the following link:
    https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning 
"""

class InputStateVariable(object):
    fuzzy_set_list = []

    def __init__(self, *args):
        self.fuzzy_set_list = args

    def get_fuzzy_sets(self):
        return self.fuzzy_set_list