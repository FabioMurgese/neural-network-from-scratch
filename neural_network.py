#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import numpy as np
import os.path
import pickle


class Layer:
    """Class implementation of a Neural Network Layer.
    """

    def __init__(self, dim, activation='sigmoid', is_output=False):
        """
        Parameters
        ----------
        dim : tuple
            the dimension of weights matrix (e.g.: ('n° previous neurons', 'n° layer's neurons') )
        activation : string
            the activation function (default: 'sigmoid')
        is_output : bool
            the flag that shows if the layer is an output layer or not
        """
        self.weight = self.__normal_weights((dim[0]+1, dim[1] if is_output else dim[1]+1))
        self.activation = activation
        self.delta = None
        self.A = None
        self.dw_old = None
        self.is_output_layer = is_output

    def __normal_weights(self, dim):
        """Initialize a matrix with normal distributed rows.
        """
        mean = 0
        variance = 0.01
        return np.random.normal(mean, variance, dim)

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
    
    def MSE(self, y, right_layer):
        """Computes the deltas using the chain rule of Mean Squared Error loss function.
        
        Parameters
        ----------
        y : numpy.array
            the target values
        right_layer : neural_network.Layer
            the next layer
        """
        if self.is_output_layer:
            n = y.shape[0]
            error = (2 / n) * (self.A - y)
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(right_layer.delta, right_layer.weight.T) * self.dZ)
        return self.delta
    
    def SSE(self, y, right_layer):
        """Computes the deltas using the chain rule of Sum of Squares Error loss function.
        
        Parameters
        ----------
        y : numpy.array
            the target values
        right_layer : neural_network.Layer
            the next layer
        """
        if self.is_output_layer:
            error = self.A - y
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(right_layer.delta, right_layer.weight.T) * self.dZ)
        return self.delta

    def BCE(self, y, right_layer):
        """Computes the deltas using the chain rule of Binary Cross Entropy loss function.
        
        Parameters
        ----------
        y : numpy.array
            the target values
        right_layer : neural_network.Layer
            the next layer
        """
        if self.is_output_layer:
            m = y.shape[0]
            error = (1 / m) * np.sum(np.maximum(self.A, 0) - self.A * y + np.log(1 + np.exp(-np.abs(self.A))))
            self.delta = np.atleast_2d(error * self.dZ)
        else:
            self.delta = np.atleast_2d(
                np.dot(right_layer.delta, right_layer.weight.T) * self.dZ)
        return self.delta
	
    def backward(self, y, right_layer, loss='mse'):
        """Perform the default loss function derivative.
        
        Parameters
        ----------
        y : numpy.array
            the target values
        right_layer : neural_network.Layer
            the next layer
        loss : string
            the loss function (default: mse)
        """
        if (loss == 'mse'):
            return self.MSE(y, right_layer)
        elif(loss == 'sse'):
            return self.SSE(y, right_layer)
        elif(loss == 'bce'):
            return self.BCE(y, right_layer)

    def update(self, lr, left_a, alpha, lmbda):
        """Update the layer weights computing delta rule.
        
        Parameters
        ----------
        lr : float
            the learning rate
        left_a : numpy.array
            the output of previous layer and the input of current layer
        alpha : float
            the momentum parameter
        lmbda : float
            the weight decay lambda regularization parameter
        """
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        dw = lr * ad
        self.dw_old = dw
        # add momentum
        if(self.dw_old is not None):
            momentum = alpha * self.dw_old
            dw += momentum
            self.weight -= dw
        else:
            self.weight -= dw
        # weight decay for regularization
        # not considering the bias
        weight_decay = lmbda * self.weight[:,1:]
        self.weight[:,1:] -= weight_decay

    
class NeuralNetwork:
    """Class implementation of an Artificial Neural Network.
    """
    
    def __init__(self, error='mee', loss='mse', learn_alg='sgd'):
        """
        Parameters
        ----------
        error : string
            the default error function (default: mee).
            - mee = Mean Euclidean Error
            - mse = Mean Squared Error
        loss : string
            the default loss function (default: sse).
            - mse = Mean Squared Error loss function
            - sse = Sum of Squares Error loss function
            - bce = Binary Cross Entropy loss function
        learn_alg : string
            the default learning algorithm (default: sgd).
            - sgd = Stochastic Gradient Descent
        """
        self.learning_algorithm = learn_alg
        self.layers = []
        self.err = error
        self.loss = loss
        
    def add(self, layer):
        """Add a new layer to the network.
        
        Parameters
        ----------
        layer : neural_network.Layer
            the new layer
        """
        self.layers.append(layer)
    
    def save(self, folder, description='', plt=None, accuracy=''):
        """Save the NeuralNetwork object to disk.
        
        Returns
        -------
        the file name
        """
        directory = os.path.join('runs', folder)
        filename = os.path.join(directory, folder+'.pkl')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        if(plt is not None):
            plt.savefig(os.path.join(directory, 'learning_curve.png'))
            plt.close()
        desc_filename = os.path.join(directory, 'description')
        with open(desc_filename, 'w') as file:
            file.write(description + '\n' + "Accuracy=" + str(accuracy))
        return filename
    
    def load(self, filename):
        """Load NeuralNetwork object from file.
        
        Returns
        -------
        the NeuralNetwork object
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def MSE(self, target, output):
        """Computes Mean Squared Error.
        
        Parameters
        ----------
        target : numpy.array
            the targer values
        output : numpy.array
            the output values of the network
        
        Returns
        -------
        the mean squared error
        """
        return float(np.square(np.array(target) - np.array(output)).mean(axis=0))
    
    def MEE(self, target, output):
        """Computes Mean Euclidean Error.
        
        Parameters
        ----------
        target : numpy.array
            the targer values
        output : numpy.array
            the output values of the network
        
        Returns
        -------
        the mean euclidean error
        """
        N = target.shape[0]
        return float(np.linalg.norm(np.array(output) - np.array(target)) / N)
    
    def error(self, target, output):
        """Computes the default error function.
        
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
        if(self.err == 'mee'):
            return self.MEE(target, output)
        elif(self.err == 'mse'):
            return self.MSE(target, output)

    def backprop(self, X, y, lr=0.1, alpha=0.5, lmbda=0.01):
        """Perform backpropagation algorithm.
        
        Parameters
        ----------
        X : numpy.array
            the inputs
        y : numpy.array
            the targets
        lr : float
            the learning rate (default: 0.1)
        alpha : float
            the momentum parameter
        lmbda : float
            the weight decay lambda regularization parameter
        
        Returns
        -------
        the error
        """
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        a=X
        # feedforward
        for layer in self.layers:
            a = layer.forward(a)
        # backward
        delta = self.layers[-1].backward(y, None)
        for l in range(len(self.layers) - 2, -1, -1):
            delta = self.layers[l].backward(delta, self.layers[l+1], self.loss)
        a = X
        # adjust weights
        for layer in self.layers:
            layer.update(lr, a, alpha, lmbda)
            a = layer.A
        return self.error(y, self.layers[-1].A)
    
    def SGD(self, training_set, validation_set, lr, epochs, mb, alpha, lmbda):
        """Executing SGD learning algorithm.
        
        Parameters
        ----------
        training_set : numpy.array
            the training set (inputs and targets)
        validation_set : numpy.array
            the validation set
        lr : float
            the learning rate
        epochs : int
            the number of epochs
        mb : int
            the mini-batch size
        alpha : float
            the momentum parameter
        lmbda : float
            the weight decay lambda regularization parameter
        
        Returns
        -------
        the training errors, the validation errors
        """
        tr_errors = []
        vl_errors = []
        for k in range(epochs):
            epoch_errors = []
            for b in range(0, len(training_set), mb):
                x = np.atleast_2d(training_set[b:b+mb,:-1]) # inputs
                y = np.atleast_2d(training_set[b:b+mb,-1]).T # targets
                error_mb = self.backprop(x, y, lr, alpha, lmbda)
                epoch_errors.append(error_mb)
            error = np.mean(epoch_errors)
            tr_errors.append(error)
            _, vl_error = self.validate(validation_set)
            vl_errors.append(vl_error)
            print(">> epoch: {:d}/{:d}, tr. error: {:f}, val. error: {:f}".format(
                    k+1, epochs, error, vl_error))
        return tr_errors, vl_errors
    
    def fit(self, training_set, validation_set, lr, epochs, mb, alpha, lmbda):
        """Computes the default learning algorithm.
        """
        if(self.learning_algorithm == 'sgd'):
            return self.SGD(training_set, validation_set, lr, epochs, mb, alpha, lmbda)
    
    def validate(self, x):
        """Computes the validation of the output of the network.
        
        Parameters
        ----------
        x : numpy.array
            the inputs
            
        Returns
        -------
        the predicted output, the default error
        """
        X = np.atleast_2d(x[:,:-1])
        y = np.atleast_2d(x[:,-1]).T
        ones = np.atleast_2d(np.ones(X.shape[0]))
        a = np.concatenate((ones.T, X), axis=1)
        for l in self.layers:
            a = l.forward(a)
        return a, self.error(y, a)
    
    def predict(self, x):
        """Computes the predicted output of the network.
        
        Parameters
        ----------
        x : numpy.array
            the inputs
            
        Returns
        -------
        the predicted output
        """
        X = np.atleast_2d(x)
        ones = np.atleast_2d(np.ones(X.shape[0]))
        a = np.concatenate((ones.T, X), axis=1)
        for l in self.layers:
            a = l.forward(a)
        return a


def k_fold_cross_validation(X, K, shuffle=True):
    """Perform k-fold cross validation splitting dataset
    in trainng set and validation set.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K, shuffle=shuffle)
    kf.get_n_splits(X)
    for tr_index, val_index in kf.split(X):
        training, validation = X[tr_index], X[val_index]
        yield training, validation
