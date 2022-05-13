import time
import numpy as np

from fuzzy.denfis.ecm import ECM
from fuzzy.self_adaptive.clip import CLIP, rule_creation


def unsupervised(train_X, trajectories, ecm=False, Dthr=1e-3, verbose=False):
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

    alpha = 0.2
    beta = 0.6

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