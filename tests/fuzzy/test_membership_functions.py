import torch
import unittest
import numpy as np
from fuzzy.common.membership_functions import Gaussian


def gaussian_np(x, center, sigma):
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


class TestGaussianMembershipFunction(unittest.TestCase):
    def test_single_input(self):
        element = 0.
        n_inputs = 1
        gaussian_mf = Gaussian(n_inputs)
        sigma = gaussian_mf.sigmas.detach().numpy()
        center = gaussian_mf.centers.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(element))
        mu_numpy = gaussian_np(element, center, sigma)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigma).all()
        assert (gaussian_mf.centers.detach().numpy() == center).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input(self):
        elements = np.random.random(10)
        n_inputs = len(elements)
        gaussian_mf = Gaussian(n_inputs)
        sigmas = gaussian_mf.sigmas.detach().numpy()
        centers = gaussian_mf.centers.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(elements))
        mu_numpy = gaussian_np(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input_with_centers_given(self):
        elements = np.random.random(5)
        n_inputs = len(elements)
        centers = np.array([0., 0.25, 0.5, 0.75, 1.0])
        gaussian_mf = Gaussian(n_inputs, centers)
        sigmas = gaussian_mf.sigmas.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(elements))
        mu_numpy = gaussian_np(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input_with_sigmas_given(self):
        elements = np.random.random(5)
        n_inputs = len(elements)
        sigmas = np.array([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        gaussian_mf = Gaussian(n_inputs, sigmas=sigmas)
        # we will now update the sigmas to be abs. value
        sigmas = np.abs(sigmas)
        centers = gaussian_mf.centers.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(elements))
        mu_numpy = gaussian_np(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input_with_both_given(self):
        elements = np.random.random(5)
        n_inputs = len(elements)
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        sigmas = np.array([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        gaussian_mf = Gaussian(n_inputs, centers=centers, sigmas=sigmas)
        # we will now update the sigmas to be abs. value
        sigmas = np.abs(sigmas)
        mu_pytorch = gaussian_mf(torch.tensor(elements))
        mu_numpy = gaussian_np(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_consistency(self):
        x = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = np.array([[9.99838713e-01, 4.31743867e-01, 2.43841232e-01, 1.16034298e-02,
                                               9.99921482e-01, 4.24180260e-01, 2.61315145e-01, 7.60780958e-04,
                                               2.29935749e-06, 9.57533721e-01, 5.72716140e-01, 1.25096634e-01,
                                               2.80023213e-03, 9.99475820e-01, 4.39181757e-01, 2.30848451e-01,
                                               7.51629638e-03]])
        centers = np.array([0.01497397, -1.3607662, 1.0883657, 1.9339248, -0.01367673,
                            2.3560243, -1.8339163, -3.3379893, -4.489564, -0.01467094,
                            -0.13278057, 0.08638719, 0.17008819, 0.01596639, -1.7408595,
                            1.5229442, 2.797653])
        sigmas = np.array([1.16553577, 1.48497267, 0.91602303, 0.91602303, 1.98733806,
                           2.53987592, 1.58646032, 1.24709336, 1.24709336, 0.10437003,
                           0.12908118, 0.08517358, 0.08517358, 1.54283158, 1.89779089,
                           1.27380911, 1.27380911])
        gaussian_mf = Gaussian(x.shape[1], centers=centers[:x.shape[1]], sigmas=sigmas[:x.shape[1]])
        mu_pytorch = gaussian_mf(torch.tensor(x[0]))

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), target_membership_degrees[0][:x.shape[1]], rtol=1e-8).all()