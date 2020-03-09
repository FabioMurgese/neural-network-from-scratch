#!/usr/bin/env python3
import numpy as np


class ActivationFunction(object):

    def function(self, x):
        """Computes activation function.

        Parameters
        ----------
        x : numpy.array
            the array of inputs

        Returns
        -------
        the activation function value
        """
        raise NotImplementedError

    def derivative(self, x):
        """Computes activation function derivative.

        Parameters
        ----------
        x : numpy.array
            the array of inputs

        Returns
        -------
        the activation function derivative value
        """
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    def function(self, x):
        """Computes sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-np.array(x)))

    def derivative(self, x):
        """Computes sigmoid function derivative.
        """
        return self.function(x) * (1 - self.function(x))


class Tanh(ActivationFunction):

    def function(self, x):
        """Computes tanh function.
        """
        return np.tanh(x)

    def derivative(self, x):
        """Computes tanh function derivative.
        """
        return 1.0 - np.tanh(x) ** 2


class ReLu(ActivationFunction):

    def function(self, x):
        """Computes ReLu function.
        """
        return np.maximum(x, 0)

    def derivative(self, x):
        """Computes ReLu function derivative.
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Linear(ActivationFunction):

    def function(self, x):
        """Computes linear function.
        """
        return x

    def derivative(self, x):
        """Computes linear function derivative.
        """
        return 1


class Softmax(ActivationFunction):

    def function(self, x):
        """Computes Softmax function.
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def derivative(self, x):
        """Computes Softmax function derivative.
        """
        return self.function(x) * (1 - self.function(x))
