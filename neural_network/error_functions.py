#!/usr/bin/env python3
import numpy as np


class ErrorFunction(object):

    def error(self, target, output):
        """Computes the error.

        Parameters
        ----------
        target : numpy.array
            the targer values
        output : numpy.array
            the output values of the network

        Returns
        -------
        the error
        """
        raise NotImplementedError


class MeanSquaredError(ErrorFunction):

    def error(self, target, output):
        """Computes Mean Squared Error.
        """
        return float(np.square(np.array(target) - np.array(output)).mean(axis=0))


class MeanEuclideanError(ErrorFunction):

    def error(self, target, output):
        """Computes Mean Euclidean Error.
        """
        sum = .0
        n = target.shape[0]
        for i in range(n):
            for j in range(target.shape[1]):
                sum += np.linalg.norm(output[i,j] - target[i,j])
        return sum / n
