#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:13:35 2021

@author: john
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Clamp(torch.autograd.Function):
    # https://discuss.pytorch.org/t/regarding-clamped-learnable-parameter/58474/3
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=1e-1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Gaussian(nn.Module):
    """
    Implementation of the Gaussian membership function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - centers: trainable parameter
        - sigmas: trainable parameter
    Examples:
        # >>> a1 = gaussian(256)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    """

    def __init__(self, in_features, centers=None, sigmas=None, trainable=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__()
        self.in_features = in_features

        # initialize centers
        if centers is None:
            self.centers = Parameter(torch.randn(self.in_features))
        else:
            self.centers = torch.tensor(centers)
            # self.centers = Parameter(torch.tensor(centers))

        # initialize sigmas
        if sigmas is None:
            self.sigmas = Parameter(torch.abs(torch.randn(self.in_features)))
        else:
            # we assume the sigmas are given to us correctly, we use the inverse of sigmoid
            # to convert the values, because later during training we need to ensure sigmas are within (0, 1)
            self.sigmas = torch.abs(torch.tensor(sigmas))
            # if trainable:
            #     a = torch.max(sigmas).item() * 2  # this is the maximum allowed sigma!!
            #     sigmas = (torch.log((1/a) * sigmas) - torch.log(1 - (1/a) * sigmas))
            # self.sigmas = Parameter(sigmas)

        self.centers.requires_grad = trainable
        self.sigmas.requiresGrad = trainable
        self.centers.grad = None
        self.sigmas.grad = None


    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # torch.sigmoid(self.sigmas) constrain the sigma values to only be (0, 1)
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))

        if not self.sigmas.requires_grad:
            return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(self.sigmas, 2)))
        else:
            clamp_class = Clamp()
            return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(clamp_class.apply(self.sigmas), 2)))

            # return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(torch.sigmoid(self.sigmas), 2)))
