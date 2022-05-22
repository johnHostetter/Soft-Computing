import torch
import numpy as np
import torch.nn as nn

from torch import optim
from torch.nn.parameter import Parameter
from membership_functions import Gaussian


class FLC(nn.Module):
    """
    Implementation of the Fuzzy Logic Controller.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - sigmas: trainable parameter
    Examples:
        # >>> antecedents = [[{'type': 'gaussian', 'parameters': {'center': 1.2, 'sigma': 0.1}},
                            {'type': 'gaussian', 'parameters': {'center': 3.0, 'sigma': 0.4}}],
                            [{'type': 'gaussian', 'parameters': {'center': 0.2, 'sigma': 0.4}}]]
        # consequences are not required, default is None
        # >>> consequences = [[{'type': 'gaussian', 'parameters': {'center': 0.1, 'sigma': 0.7}},
                            {'type': 'gaussian', 'parameters': {'center': 0.4, 'sigma': 0.41}}],
                            [{'type': 'gaussian', 'parameters': {'center': 0.9, 'sigma': 0.32}}]]
        # if consequences are not to be specified, leave the key-value out
        # >>> rules = [{'antecedents':[0, 0], 'consequences':[0]}, {'antecedents':[1, 0], 'consequences':[1]}]
        # >>> n_input = len(antecedents)  # the length of antecedents should be equal to number of inputs
        # >>> n_output = len(consequences)  # the length of antecedents should be equal to number of inputs
        # >>> flc = FLC(n_input, n_output, antecedents, rules, consequences)
        # >>> x = torch.randn(n_input)
        # >>> y = flc(x)
    """

    def __init__(self, in_features, out_features, antecedents, rules, consequences=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - consequences: (optional) trainable parameter
            consequences are initialized randomly by default,
            but sigmas must be > 0
        """
        super(FLC, self).__init__()
        self.in_features = in_features

        # find the number of antecedents per input variable
        num_of_antecedents = np.zeros(in_features).astype('int32')
        unique_id = 0
        gaussians = {'centers': [], 'sigmas': []}  # currently, we assume only Gaussians are used
        self.input_variable_ids = []
        self.transformed_x_length = 0
        for input_variable_idx in range(in_features):
            num_of_antecedents[input_variable_idx] = len(antecedents[input_variable_idx])
            self.input_variable_ids.append(set())
            for term_idx, antecedent in enumerate(antecedents[input_variable_idx]):
                gaussians['centers'].append(antecedent['parameters']['center'])
                gaussians['sigmas'].append(antecedent['parameters']['sigma'])
                antecedent['id'] = unique_id
                self.input_variable_ids[-1].add(unique_id)
                unique_id += 1
        self.transformed_x_length = unique_id

        # find the total number of antecedents across all input variables
        n_rules = len(rules)
        self.links_between_antecedents_and_rules = np.zeros((num_of_antecedents.sum(), n_rules))

        for rule_idx, rule in enumerate(rules):
            for input_variable_idx, term_idx in enumerate(rule['antecedents']):
                new_term_idx = antecedents[input_variable_idx][term_idx]['id']
                self.links_between_antecedents_and_rules[new_term_idx, rule_idx] = 1

        # begin creating the model's layers
        self.input_terms = Gaussian(in_features=self.in_features, centers=gaussians['centers'],
                                    sigmas=gaussians['sigmas'])

        # initialize consequences
        if consequences is None:
            num_of_consequent_terms = len(rules)
            self.consequences = Parameter(torch.randn(num_of_consequent_terms, out_features))
        else:
            self.consequences = Parameter(torch.tensor(consequences))

        self.consequences.requires_grad = True

    def __transform(self, x):
        """
        Transforms the given 'x' to make it compatible with the first layer.
        """
        shape = x.shape
        n_observations = shape[0]  # number of observations
        # num_of_input_variables = shape[1]
        new_x = np.zeros((n_observations, self.transformed_x_length))
        for input_variable_idx, indices_to_repeat_for in enumerate(self.input_variable_ids):
            min_column_idx = min(indices_to_repeat_for)
            max_column_idx = max(indices_to_repeat_for) + 1
            copies = len(indices_to_repeat_for)  # how many copies we should make of this column
            new_x[:, min_column_idx:max_column_idx] = np.repeat(x[:, input_variable_idx], copies).reshape(
                (n_observations, copies))
        return torch.tensor(new_x)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # we need to make the given x compatible with our first layer,
        # which means repeating it for some entries
        antecedents_memberships = self.input_terms(self.__transform(x))
        # (antecedents_memberships.detach().numpy()[:, :, np.newaxis] * flc.links_between_antecedents_and_rules)
        terms_to_rules = antecedents_memberships[:, :, None] * torch.tensor(self.links_between_antecedents_and_rules)
        terms_to_rules[terms_to_rules == 0] = 1.0  # ignore zeroes, this is from the weights between terms and rules
        # the shape of terms_to_rules is (num of observations, num of ALL terms, num of rules)
        rules_applicability = terms_to_rules.prod(dim=1)
        numerator = torch.matmul(rules_applicability, self.consequences.double())
        # numerator = (rules_applicability * self.consequences).sum(dim=1)
        denominator = rules_applicability.sum(dim=1)
        # return numerator / denominator  # the dim=1 is taking product across ALL terms, now shape (num of observations, num of rules)
        return numerator / denominator[:, None]  # shape is (num of observations, num of actions)


# helper function to train a model
def train_model(model, X, actual_y):
    '''
    Function trains the model and prints out the training log.
    INPUT:
        model - initialized PyTorch model ready for training.
        trainloader - PyTorch dataloader for training data.
    '''
    # setup training

    # define loss function
    criterion = nn.MSELoss()
    # define learning rate
    learning_rate = 0.003
    # define number of epochs
    epochs = 20
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # run training and print out the loss to make sure that we are actually fitting to the training set
    print('Training the model. Make sure that loss decreases after each epoch.\n')

    losses = []
    num_of_epochs = 0
    while (len(losses) > 1 and losses[-2] > losses[-1]) or num_of_epochs < 1000:
        # print(num_of_epochs)
        predicted_y = model(X)
        loss = criterion(predicted_y, actual_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        num_of_epochs += 1
        losses.append(loss.item())
    print(losses[-1])


if __name__ == '__main__':
    x = np.array([[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]])
    # actual_y = np.array([1.5, 0.6, 0.9, 0.7, 1.3])
    # actual_y = torch.tensor(actual_y)
    actual_y = torch.randn(5, 2).double()
    print('actual y')
    print(actual_y)

    antecedents = [[{'type': 'gaussian', 'parameters': {'center': 1.2, 'sigma': 0.1}},
                    {'type': 'gaussian', 'parameters': {'center': 3.0, 'sigma': 0.4}}],
                   [{'type': 'gaussian', 'parameters': {'center': 0.2, 'sigma': 0.4}},
                    {'type': 'gaussian', 'parameters': {'center': 0.6, 'sigma': 0.4}}]]
    # consequences are not required, default is None
    consequences = [[{'type': 'gaussian', 'parameters': {'center': 0.1, 'sigma': 0.7}},
                     {'type': 'gaussian', 'parameters': {'center': 0.4, 'sigma': 0.41}}],
                    [{'type': 'gaussian', 'parameters': {'center': 0.9, 'sigma': 0.32}}]]
    # if consequences are not to be specified, leave the key-value out
    rules = [{'antecedents': [0, 0], 'consequences': [0]}, {'antecedents': [1, 0], 'consequences': [1]},
             {'antecedents': [1, 1], 'consequences': [1]}]
    n_input = len(antecedents)  # the length of antecedents should be equal to number of inputs
    n_output = len(consequences)  # the length of antecedents should be equal to number of inputs
    flc = FLC(n_input, n_output, antecedents, rules, None)
    # x = torch.randn(n_input)
    predicted_y = flc(x)
    print(predicted_y)

    print('sigmas')
    print(flc.input_terms.sigmas)
    print('centers')
    print(flc.input_terms.centers)
    print('consequences')
    print(flc.consequences)

    train_model(flc, x, actual_y)

    print('after training')

    new_predicted_y = flc(x)
    print(new_predicted_y)
    print('sigmas')
    print(flc.input_terms.sigmas)
    print('centers')
    print(flc.input_terms.centers)
    print('consequences')
    print(flc.consequences)
