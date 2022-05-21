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
