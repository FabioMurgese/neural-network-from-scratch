#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import numpy as np


class Layer:
    """Class implementation of a Neural Network Layer.
    """
    
    def __init__(self, dim, activation='sigmoid', is_output=False):
        """
        Parameters
        ----------
        dim : tuple
            the dimension of weights matrix
        activation : string
            the activation function (default: 'sigmoid')
        is_output : bool
            the flag that shows if the layer is an output layer or not
        """
        self.weight = self.__normal_distr_weights_init((dim[0]+1, dim[1] if is_output else dim[1]+1))
        self.activation = activation
        self.delta = None
        self.A = None
        self.is_output_layer = is_output
        
    def __normal_distr_weights_init(self, dim):
        """Initialize a matrix with normal distributed rows.
        """
        return 2 * np.random.normal(0, 1, dim) - 1
    
    def __str__(self):
        return '''Layer ========
    weight: 
        {0}
    delta: 
        {1}
    output:
        {2}
    activation:
        {3}
    is output:
        {4}
=============='''.format(str(self.weight), self.delta, self.A, self.activation, self.is_output_layer)
    
    def __sigmoid(self, x):
        """Computes sigmoid function.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return 1.0 / (1.0 + np.exp(-np.array(x)))
    
    def __sigmoid_prime(self, x):
        """Computes sigmoid function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    def __tanh(self, x):
        """Computes tanh function.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return np.tanh(x)

    def __tanh_prime(self, x):
        """Computes tanh function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return 1.0 - np.tanh(x) ** 2
    
    def __relu(self, x):
        """Computes relu function.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return np.maximum(x, 0)

    def __relu_prime(self, x):
        """Computes relu function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def activation_function(self, x):
        """Computes the default activation function.
        
        Parameters
        ----------
        x : numpy.array
            the input vector
        """
        if(self.activation == 'sigmoid'):
            return self.__sigmoid(x)
        elif(self.activation == 'tanh'):
            return self.__tanh(x)
        elif(self.activation == 'relu'):
            return self.__relu(x)
        else:
            return
    
    def activation_function_prime(self, x):
        """Computes the default activation function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the input vector
        """
        if(self.activation == 'sigmoid'):
            return self.__sigmoid_prime(x)
        elif(self.activation == 'tanh'):
            return self.__tanh_prime(x)
        elif(self.activation == 'relu'):
            return self.__relu_prime(x)
        else:
            return
    
    def forward(self, x):
        """Computes the output of the layer.
        
        Parameters
        ----------
        x : numpy.array
            the inputs
            
        Returns
        -------
        the output of the layer
        """
        z = np.dot(x, self.weight) # the net of the layer
        self.A = self.activation_function(z) # sigma(net)
        self.dZ = np.atleast_2d(self.activation_function_prime(z)) # partial derivative
        return self.A
        
    def backward(self, y, right_layer):
        """Computes the deltas by the chain rule.
        
        Parameters
        ----------
        y : numpy.array
            the target values
        right_layer : Layer
            the next layer
        """
        if self.is_output_layer:
            error = self.A - y
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(right_layer.delta, right_layer.weight.T) * self.dZ)
        return self.delta
    
    def update(self, lr, left_a):
        """Update the layer weights computing delta rule.
        
        Parameters
        ----------
        lr : float
            the learning rate
        left_a : numpy.array
            the output of previous layer and the input of current layer
        """
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        self.weight -= lr * ad

    
class NeuralNetwork:
    """Class implementation of an Artificial Neural Network.
    """
    
    def __init__(self):
        """Initialize a neural network.
        """
        self.layers = []
#        for i in range(1, len(dims) - 1):
#            dim = (dims[i - 1] + 1, dims[i] + 1)
#            self.layers.append(Layer(id=i, dim=dim))
#        dim = (dims[i] + 1, dims[i + 1])
#        self.layers.append(Layer(id=len(dims) - 1, dim=dim, is_output=True))
        
    def __str__(self):
        return('''Network ==============
{0}
========================''').format('\n\n'.join(
            [str(layer) for layer in self.layers]))
        
    def add(self, layer):
        """Add a new layer to the network.
        
        Parameters
        ----------
        layer : neural_network.Layer
            the new layer
        """
        self.layers.append(layer)

    def backprop(self, X, y, lr=0.1):
        """Perform backpropagation algorithm.
        
        Parameters
        ----------
        X : numpy.array
            the inputs
        y : numpy.array
            the targets
        lr : float
            the learning rate (default: 0.1)
        
        Returns
        -------
        the mean square error
        """
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        a=X
        # feedforward
        for l in range(len(self.layers)):
            a = self.layers[l].forward(a)
        delta = self.layers[-1].backward(y, None)
        #backward
        for l in range(len(self.layers) - 2, -1, -1):
            delta = self.layers[l].backward(delta, self.layers[l+1])
        a = X
        # adjust weights
        for layer in self.layers:
            layer.update(lr, a)
            a = layer.A
        return float(np.square(np.array(y) - np.array(self.layers[-1].A)).mean(axis=0))
    
    def fit(self, X, y, lr, epochs):
        """Executing learning algorithm for a certain time of epochs.
        
        Parameters
        ----------
        X : numpy.array
            the input matrix
        y : numpy.array
            the target vector
        lr : float
            the learning rate
        epochs : int
            the number of epochs
        
        Returns
        -------
        the errors of the learning algorithm at each epoch
        """
        errors = []
        for k in range(epochs):
            error = self.backprop(X, y, lr=lr)
            errors.append(error)
            print(">> epoch: {:d}/{:d}, error: {:f}".format(k+1, epochs, error))
        return errors
    
    def predict(self, x):
        """Computes the predicted output of the network.
        
        Parameters
        ----------
        x : list
            the list of inputs
            
        Returns
        -------
        the predicted output
        """
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.layers)):
            a = self.layers[l].forward(a)
        return a
