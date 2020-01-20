import numpy as np

class ActivationFunction():

    def compute(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()

class Sigmoid(ActivationFunction):

    def compute(self, x):
        """Computes sigmoid function.

        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return 1.0 / (1.0 + np.exp(-np.array(x)))

    def derivative(self, x):
        """Computes sigmoid function derivative.

        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return self.compute(x) * (1 - self.compute(x))