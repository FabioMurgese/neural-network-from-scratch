#!/usr/bin/env python3
import numpy as np


class Regularizer(object):
    
    def regularize(self, w):
        """Regularize the weight matrix.
        
        Parameters
        ----------
        lmbda : float
            the lambda parameter
        w : numpy.array
            tha weight matrix of a layer
        """
        raise NotImplementedError


class L2(Regularizer):
    
    def __init__(self, lmbda=1e-4):
        self.lmbda = lmbda
    
    def regularize(self, w):
        """Computes Tikhonov regularization (L2).
        """
        # computes weight decay not considering the bias
        weight_decay = 2 * self.lmbda * w[:,1:]
        w[:,1:] -= weight_decay
        return w
