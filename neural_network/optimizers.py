#!/usr/bin/env python3
import numpy as np


class Optimizer(object):
    
    def train(self, training_set, validation_set, net):
        """Train the network.
        
        Parameters
        ----------
        training_set : numpy.array
            the training set (inputs and targets)
        validation_set : numpy.array
            the validation set
        net : neural_network.NeuralNetwork
        
        Returns
        -------
        the training errors, the validation errors, the trained network
        """
        raise NotImplementedError


class SGD(Optimizer):
    
    def __init__(self, lr=0.1, epochs=100, mb=20, alpha=0.2):
        """
        Parameters
        ----------
        lr : float
            the learning rate
        epochs : int
            the number of epochs
        mb : int
            the mini-batch size
        alpha : float
            the momentum parameter
        """
        self.lr = lr
        self.epochs = epochs
        self.mb = mb
        self.alpha = alpha
    
    def __backpropagation(self, X, y, net):
        """Perform backpropagation algorithm.
        
        Parameters
        ----------
        X : numpy.array
            the inputs
        y : numpy.array
            the targets
        net : neural_network.NeuralNetwork
            the neural network
        
        Returns
        -------
        the error of the network
        """
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        a=X
        # feedforward
        for layer in net.layers:
            a = layer.forward(a)
        # backward
        delta = net.layers[-1].backward(y, None, net.loss)
        for l in range(len(net.layers) - 2, -1, -1):
            delta = net.layers[l].backward(delta, net.layers[l+1], net.loss)
        a = X
        # adjust weights
        for layer in net.layers:
            layer.update(self.lr, a, self.alpha, net.regularizer)
            a = layer.A
        return net.err.error(y, net.layers[-1].A)
    
    def train(self, training_set, validation_set, net):
        """Executing SGD learning algorithm.
        """        
        n_outputs = net.layers[-1].weight.shape[1]
        tr_errors = []
        vl_errors = []
        for k in range(self.epochs):
            epoch_errors = []
            for b in range(0, len(training_set), self.mb):
                x = np.atleast_2d(training_set[b:b+self.mb, :-n_outputs])
                y = np.atleast_2d(training_set[b:b+self.mb, -n_outputs:])
                error_mb = self.__backpropagation(x, y, net)
                epoch_errors.append(error_mb)
            error = np.mean(epoch_errors)
            tr_errors.append(error)
            _, vl_error = net.validate(validation_set)
            vl_errors.append(vl_error)
            print(">> epoch: {:d}/{:d}, tr. error: {:f}, val. error: {:f}".format(
                    k+1, self.epochs, error, vl_error))
        return tr_errors, vl_errors, net
        

class Adam(Optimizer):
    
    def __init__(self, alpha=0.01, epochs=100):
        self.alpha = alpha
        self.epochs = epochs
    
    def train(self, training_set, validation_set, net):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        