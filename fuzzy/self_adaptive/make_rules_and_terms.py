import time
import torch
import numpy as np

from fuzzy.denfis.ecm import ECM
from fuzzy.self_adaptive.clip import CLIP, rule_creation


def calc_flc_outputs(self, X):
    flc_outputs = []
    for flc_idx, flc in enumerate(self.flcs):
        flc_outputs.append([])
        for training_idx, x in enumerate(X):
            flc_outputs[flc_idx].append(flc.legacy_predict(x))

    return np.array(flc_outputs)


def calc_offline_loss(self, flc_outputs, y, action_indices):
    target_qvalues = torch.tensor(y, dtype=torch.float32)
    action_indices = torch.tensor(action_indices, dtype=torch.int64)
    pred_qvalues = torch.tensor(flc_outputs, dtype=torch.float32)
    logsumexp_qvalues = torch.logsumexp(pred_qvalues, dim=-1)

    tmp_pred_qvalues = pred_qvalues.gather(
        1, action_indices.reshape(-1, 1)).squeeze()
    cql_loss = logsumexp_qvalues - tmp_pred_qvalues

    loss = torch.mean((pred_qvalues - target_qvalues) ** 2)
    loss = loss + self.cql_alpha * torch.mean(cql_loss)
    return loss


def offline_update(self, train_X, train_y, train_action_indices, val_X, val_y, val_action_indices):
    # calculate the outputs

    train_flc_outputs = self.predict(train_X).numpy()
    val_flc_outputs = self.predict(val_X).numpy()

    # calculate the loss for reporting & monitoring

    train_loss = self.calc_offline_loss(train_flc_outputs, train_y, train_action_indices)
    val_loss = self.calc_offline_loss(val_flc_outputs, val_y, val_action_indices)

    # gradient descent

    for flc_idx, flc in enumerate(self.flcs):
        if train_y.ndim == 1:  # single observation
            y_ = train_y[flc_idx]
        elif train_y.ndim == 2:  # multiple observations
            y_ = train_y[:, flc_idx]

        constant = 2 / len(y_)
        loss_on_train_data = (flc.predict(train_X) - y_[:, np.newaxis])
        mse = np.zeros(self.flcs[0].M)
        offlines = np.zeros(self.flcs[0].M)

        for l in range(flc.M):
            z = flc.z(l)
            # consequent terms' centers
            b = flc.b()
            # mse[l] = float(constant * (loss_on_train_data * (z / b)[:, np.newaxis]))
            mse[l] = float((loss_on_train_data.T[0] * (z / b)).sum() * constant)

            # correct offline
            numerator = (np.exp(train_flc_outputs[:, flc_idx]) * z / b)
            denominator = (np.log(2) * np.exp(train_flc_outputs)).sum(axis=1)
            offlines[l] = (((numerator / denominator) - (z / b)).mean())

        offline_adjustment = self.cql_alpha * offlines
        # learning_rate = 6.25e-05
        flc.y, _ = self.adam.update(self.timestep, w=flc.y, dw=(offline_adjustment + mse))
        # flc.y -= learning_rate * (offline_adjustment + mse)
    return train_loss, val_loss


# Yield successive n-sized chunks from l.
def divide_chunks(lst, n):
    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_data_from_trajectories(self, batch, gamma):
    batch = np.array(batch)
    states = np.array([list(state) for state in batch[:, 0]])
    next_states = np.array([list(next_state) for next_state in batch[:, 2]])
    action_indices = batch[:, 1].astype('int32')
    q_values = self.predict(states)
    q_values_next = self.predict(next_states)
    rewards = batch[:, 3].astype('float32')
    dones = batch[:, 4]
    q_values[torch.arange(len(q_values)), action_indices] = torch.tensor(rewards)
    max_q_values_next, max_q_values_next_indices = torch.max(q_values_next, dim=1)
    max_q_values_next *= gamma
    q_values[torch.arange(len(q_values)), action_indices] += torch.tensor(
        [0.0 if done else float(max_q_values_next[idx]) for idx, done in enumerate(dones)])
    targets = q_values.numpy()

    return states, targets, action_indices


def replay(self, train_memory, size, validation_memory, gamma=0.9, online=True):
    """ Add experience replay to the DQN network class. """
    # Make sure the memory is big enough
    if len(train_memory) >= size:
        # Sample a batch of experiences from the agent's memory
        # batch = random.sample(train_memory, size)

        # divide the memory into batches of length 'size'
        batches = list(divide_chunks(train_memory, size))

        train_losses = []
        val_losses = []
        for batch in batches:
            self.timestep += 1  # the number of gradient descent updates performed thus far

            # Extract information from the data
            train_states, train_targets, train_action_indices = self.extract_data_from_trajectories(batch, gamma)

            val_states, val_targets, val_action_indices = self.extract_data_from_trajectories(validation_memory,
                                                                                              gamma)

            if online:  # needs updating (to remove validation data arguments)
                train_loss, val_loss = self.offline_update(train_states, train_targets, train_action_indices,
                                                           val_states, val_targets, val_action_indices)
            else:
                train_loss, val_loss = self.offline_update(train_states, train_targets, train_action_indices,
                                                           val_states, val_targets, val_action_indices)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return np.array(train_losses), np.array(val_losses)


def unsupervised(train_X, trajectories, alpha=0.2, beta=0.6, ecm=False, Dthr=1e-3, verbose=False):
    """
    Trains the CFQLModel with its AdaptiveNeuroFuzzy object on the provided training data, 'train_X',
    and their corresponding trajectories.

    Parameters
    ----------
    train_X : 2-D Numpy array
        The input vector, has a shape of (number of observations, number of inputs/attributes).
    trajectories : list
        A list containing elements that have the form of (state, action, reward, next state, done).
        The 'state' and 'next state' items are 1D Numpy arrays that have the shape of (number of inputs/attributes,).
        The 'action' item is an integer that references the index of the action chosen when in 'state'.
        The 'reward' item is a float that describes the immediate reward received after taking 'action' in 'state'.
        The 'done' item is a boolean that is True if this list element is the end of an episode, False otherwise.
    ecm : boolean, optional
        This boolean controls whether to enable the ECM algorithm for candidate rule generation. The default is False.
    Dthr : float, optional
        The distance threshold for the ECM algorithm; only matters if ECM is enabled. The default is 1e-3.
    verbose : boolean, optional
        If enabled (True), the execution of this function will print out step-by-step to show progress. The default is False.

    Returns
    -------
    None.

    """
    print('The shape of the training data is: (%d, %d)\n' %
          (train_X.shape[0], train_X.shape[1]))
    train_X_mins = train_X.min(axis=0)
    train_X_maxes = train_X.max(axis=0)

    # this Y array only exists to make the rule generation simpler
    dummy_Y = np.zeros(train_X.shape[0])[:, np.newaxis]
    Y_mins = np.array([-1.0])
    Y_maxes = np.array([1.0])

    if verbose:
        print('Creating/updating the membership functions...')

    start = time.time()
    antecedents = CLIP(train_X, dummy_Y, train_X_mins, train_X_maxes,
                       [], alpha=alpha, beta=beta)
    end = time.time()
    if verbose:
        print('membership functions for the antecedents generated in %.2f seconds.' % (
                end - start))

    start = time.time()
    consequents = CLIP(dummy_Y, train_X, Y_mins, Y_maxes, [],
                       alpha=alpha, beta=beta)
    end = time.time()
    if verbose:
        print('membership functions for the consequents generated in %.2f seconds.' % (
                end - start))

    if ecm:
        if verbose:
            print('\nReducing the data observations to clusters using ECM...')
        start = time.time()
        clusters = ECM(train_X, [], Dthr)
        if verbose:
            print('%d clusters were found with ECM from %d observations...' % (
                len(clusters), train_X.shape[0]))
        reduced_X = [cluster.center for cluster in clusters]
        reduced_dummy_Y = dummy_Y[:len(reduced_X)]
        end = time.time()
        if verbose:
            print('done; the ECM algorithm completed in %.2f seconds.' %
                  (end - start))
    else:
        reduced_X = train_X
        reduced_dummy_Y = dummy_Y

    if verbose:
        print('\nCreating/updating the fuzzy logic rules...')
    start = time.time()
    antecedents, consequents, rules, weights = rule_creation(reduced_X, reduced_dummy_Y,
                                                             antecedents,
                                                             consequents,
                                                             [],
                                                             [],
                                                             problem_type='SL',
                                                             consistency_check=False)

    K = len(rules)
    end = time.time()
    if verbose:
        print('%d fuzzy logic rules created/updated in %.2f seconds.' %
              (K, end - start))
    return rules, weights, antecedents, consequents
