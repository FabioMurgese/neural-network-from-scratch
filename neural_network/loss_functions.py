#!/usr/bin/env python3
import numpy as np


class LossFunction(object):

    def delta(self, layer, right_layer, target):
        """Computes the partial derivatives of the Layer.

        Parameters
        ----------
        layer : neural_network.Layer
            the neural network layer
        right_layer : neural_network.Layer
            the next layer
        target : numpy.array
            the targets

        Returns
        -------
        the deltas
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Class implementation of Mean Squared Error loss function.
    """
    
    def delta(self, layer, right_layer, target):
        if layer.is_output_layer:
            n = target.shape[0]
            output = layer.output
            error = (2 / n) * (output - target)
            return np.atleast_2d(error * layer.dz)
        else:
            return np.atleast_2d(np.dot(right_layer.delta, right_layer.weight.T) * layer.dz)
    
    
class SumSquaresError(LossFunction):
    """Class implementation of Sum of Squares Error loss function.
    """
    
    def delta(self, layer, right_layer, target):
        if layer.is_output_layer:
            output = layer.output
            error = output - target
            return np.atleast_2d(error * layer.dz)
        else:
            return np.atleast_2d(np.dot(right_layer.delta, right_layer.weight.T) * layer.dz)
        

class BinaryCrossEntropy(LossFunction):
    """Class implementation of Binary Cross Entropy loss function.
    """
    
    def delta(self, layer, right_layer, target):
        if layer.is_output_layer:
            m = target.shape[0]
            output = layer.output
            error = -((1 / m) * np.sum(np.maximum(output, 0) - output * target + np.log(1 + np.exp(-np.abs(output)))))
            return np.atleast_2d(error * layer.dz)
        else:
            return np.atleast_2d(np.dot(right_layer.delta, right_layer.weight.T) * layer.dz)
