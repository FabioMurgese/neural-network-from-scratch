#!/usr/bin/env python3
import numpy as np


class Optimizer(object):
    
    def train(self, training_set, validation_set, net, verbose=False):
        """Train the network.
        
        Parameters
        ----------
        training_set : numpy.array
            the training set (inputs and targets)
        validation_set : numpy.array
            the validation set
        net : neural_network.NeuralNetwork
            the neural network to train
        
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
        # feedforward
        net.feedforward(X)
        # backward
        net.compute_deltas(y)
        # update weights
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        a = X
        for layer in net.layers:
            x = np.atleast_2d(a)
            d = np.atleast_2d(layer.delta)
            dw = self.lr * x.T.dot(d)
            layer.dw_old = dw
            # add momentum
            if(layer.dw_old is not None):
                momentum = self.alpha * layer.dw_old
                dw += momentum
                layer.weight -= dw
            else:
                layer.weight -= dw
            # perform regularization
            if(net.regularizer is not None):
                layer.weight = net.regularizer.regularize(layer.weight)
            a = layer.A
        return net.err.error(y, net.layers[-1].A)
    
    def train(self, training_set, validation_set, net, compute_accuracy=False, verbose=False):
        """Executes Stochastic Gradient Descent learning algorithm (with momentum).
        """
        tr_errors = []
        vl_errors = []
        tr_accuracy = []
        vl_accuracy = []
        for k in range(self.epochs):
            epoch_errors = []
            for b in range(0, len(training_set), self.mb):
                x = np.atleast_2d(training_set[b:b+self.mb, :-net.n_outputs()])
                y = np.atleast_2d(training_set[b:b+self.mb, -net.n_outputs():])
                error_mb = self.__backpropagation(x, y, net)
                epoch_errors.append(error_mb)
            tr_error = np.mean(epoch_errors)
            tr_errors.append(tr_error)
            _, vl_error = net.validate(validation_set)
            vl_errors.append(vl_error)
            if verbose:
                print(">> epoch: {:d}/{:d}, tr. error: {:f}, val. error: {:f}".format(
                        k+1, self.epochs, tr_error, vl_error))
            if compute_accuracy:
                tr_accuracy.append(net.compute_accuracy(training_set[:,-net.n_outputs():], net.predict(training_set[:, :-net.n_outputs()])))
                vl_accuracy.append(net.compute_accuracy(validation_set[:, -net.n_outputs():], net.predict(validation_set[:, :-net.n_outputs()])))
        return tr_errors, vl_errors, tr_accuracy, vl_accuracy, net
        

class Adam(Optimizer):
    """
    https://arxiv.org/pdf/1412.6980v8.pdf
    """
    
    def __init__(self, alpha=0.01, epochs=100):
        self.alpha = alpha
        self.epochs = epochs
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
    
    def train(self, training_set, validation_set, net, compute_accuracy=False, verbose=False):
        """Executes Adam learning algorithm.
        """
        tr_errors = []
        vl_errors = []
        tr_accuracy = []
        vl_accuracy = []
        X = np.atleast_2d(training_set[:, :-net.n_outputs()])
        y = np.atleast_2d(training_set[:, -net.n_outputs():])
        m = [0] * len(net.layers) # first moments
        v = [0] * len(net.layers) # second moments
        for t in range(1, self.epochs+1):
            # feedforward
            net.feedforward(X)
            # computes deltas
            net.compute_deltas(y)
            # adjust weights
            ones = np.atleast_2d(np.ones(X.shape[0]))
            x = np.concatenate((ones.T, X), axis=1)
            # apply Adam weight update for each layer
            for i, layer in enumerate(net.layers):
                # the gradient of the layer
                g = np.atleast_2d(x.T.dot(layer.delta))
                # update biased first moment estimate. 
                # the 1st moment is the exponential average of gradients along weights.
                m[i] = self.beta_1 * m[i] + (1 - self.beta_1) * g
                # update biased second raw moment estimate.
                # the 2nd moment is the exponential average of squares of gradients along weights
                v[i] = self.beta_2 * v[i] + (1 - self.beta_2) * (g ** 2)
                # compute bias-corrected first moment estimate
                m_hat = m[i] / (1 - (self.beta_1 ** t))
                # compute bias-corrected second raw moment estimate
                v_hat = v[i] / (1 - (self.beta_2 ** t))
                # update parameters
                dw = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
                layer.weight -= dw
                x = layer.A
            tr_error = net.err.error(y, net.layers[-1].A)
            tr_errors.append(tr_error)
            _, vl_error = net.validate(validation_set)
            vl_errors.append(vl_error)
            if verbose:
                print(">> epoch: {:d}/{:d}, tr. error: {:f}, val. error: {:f}".format(
                        t, self.epochs, tr_error, vl_error))
            if compute_accuracy:
                tr_accuracy.append(net.compute_accuracy(training_set[:,-net.n_outputs():], net.predict(training_set[:, :-net.n_outputs()])))
                vl_accuracy.append(net.compute_accuracy(validation_set[:, -net.n_outputs():], net.predict(validation_set[:, :-net.n_outputs()])))
        return tr_errors, vl_errors, tr_accuracy, vl_accuracy, net
