#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import numpy as np


class Layer:
    """Class implementation of a Neural Network Layer.
    """
    
    def __init__(self, n_neurons=2, n_inputs=2, bias=None, activation='sigmoid'):
        """
        Parameters
        ----------
        n_neurons : int
            the numbers of neurons (default: 2)
        n_inputs : int
            the number of inputs for each neuron (default: 2)
        bias : float
            the bias
        activation : string
            the activation function (default: 'sigmoid')
        """
        self.activation = activation
        self.bias = bias if bias is not None else np.random.random()
        self.weights = self.__normal_distr_weights_init(
                rows=n_neurons, cols=n_inputs+1) # +1 for the bias
        self.deltas = np.zeros((n_neurons, 1))
        self.outputs = np.zeros((n_neurons, 1))
        self.inputs = np.zeros((n_inputs, 1))
        
    def __normal_distr_weights_init(self, rows, cols):
        """Initialize a matrix with normal distributed rows.
        """
        return np.random.normal(0, 1, (rows, cols))
    
    def __str__(self):
        return '''Layer ========
    weights: 
        {0}
    deltas: 
        {1}
    inputs:
        {2}
    outputs:
        {3}
    bias:
        {4}
    activation:
        {5}
=============='''.format(str(self.weights), self.deltas, self.inputs, self.outputs, self.bias, self.activation)
    
    def __sigmoid(self, x):
        """Compute sigmoid function.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return 1.0 / (1.0 + np.exp(-np.array(x)))
    
    def __sigmoid_prime(self, x):
        """Compute sigmoid function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the array of inputs
        """
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    
    def activation_function(self, x):
        """Compute the default activation function.
        
        Parameters
        ----------
        x : numpy.array
            the input vector
        """
        if(self.activation == 'sigmoid'):
            return self.__sigmoid(x)
        else:
            return
    
    def activation_function_prime(self, x):
        """Compute the default activation function derivative.
        
        Parameters
        ----------
        x : numpy.array
            the input vector
        """
        if(self.activation == 'sigmoid'):
            return self.__sigmoid_prime(x)
        else:
            return
    
    def feedforward(self, x):
        """Compute the output of the layer.
        
        Parameters
        ----------
        x : numpy.array
            the inputs
            
        Returns
        -------
        the output of the layer
        """
        x.insert(0, self.bias)
        x = np.array(x)
        x.reshape(x.shape[0], 1)
        self.inputs = x
        self.outputs = list(self.activation_function(self.weights.dot(self.inputs)))
        return self.outputs
    
    def adjust_weights(self, lr):
        """Adjust the layer weights computing delta rule.
        
        Parameters
        ----------
        lr : float
            the learning rate
        """
        inputs = self.inputs.reshape(self.inputs.shape[0], 1).T
        deltas = np.multiply(lr, self.deltas)
        deltas = np.dot(deltas, inputs)
        self.weights = self.weights + deltas
    
    
class NeuralNetwork:
    """Class implementation of an Artificial Neural Network.
    """
    
    def __init__(self, n_inputs=2, sizes=[3,3,1]):
        """Initialize a neural network.
        
        Parameters
        ----------
        n_inputs : int
            the number of inputs (default: 2)
        sizes : list
            the dimensions of the network (default: [3,3,1])
        """
        assert len(sizes) > 0, "Network must have sizes!"
        self.layers = []
        for i in range(len(sizes)):
            self.layers.append(Layer(
                    bias=1,
                    n_neurons=sizes[i], 
                    n_inputs=(n_inputs) if i == 0 else (sizes[i-1]) ))
    
    def __str__(self):
        return('''Network ==============
{0}
========================''').format('\n\n'.join(
            [str(layer) for layer in self.layers]))

    def backprop(self, training_examples, lr=0.5):
        """Backpropagation algorithm.
        
        Parameters
        ----------
        training_examples : list
            the list of examples
        lr : float
            the learning rate (default: 0.5)
        
        Returns
        -------
        the mean square error of the mini batch
        """
        for example in training_examples:
            x = example[:-1]
            y = example[-1]
            # propagate the inputs forward the network
            for i, l in enumerate(self.layers):
                inputs = x if i == 0 else self.layers[i-1].outputs
                l.feedforward(inputs)
            # compute deltas for each layer
            for i in range(len(self.layers), 0, -1):
                current_layer = self.layers[i-1]
                out = current_layer.outputs
                # chain rule
                if i == len(self.layers):
                    current_layer.deltas = np.dot(
                            current_layer.activation_function_prime(out), 
                            np.subtract(y, out))
                else:
                    succ_layer = self.layers[i]
                    weighted_errors = np.dot(succ_layer.weights.T, succ_layer.deltas)
                    weighted_errors = weighted_errors.reshape(weighted_errors.shape[0], 1)
                    current_layer.deltas = np.dot(
                             current_layer.activation_function_prime(out), 
                             weighted_errors)
            # adjust the weights of the neurons of each layer
            for l in self.layers:
                l.adjust_weights(lr)
        return (np.square(np.array(y) - np.array(self.layers[-1].outputs))).mean(axis=0)
    
    def predict(self, x):
        """Compute the predicted output of the network.
        
        Parameters
        ----------
        x : list
            the list of inputs
            
        Returns
        -------
        the predicted output
        """
        output = None
        for i, l in enumerate(self.layers):
            inputs = x if i == 0 else self.layers[i-1].outputs
            output = l.feedforward(inputs)
        return output
