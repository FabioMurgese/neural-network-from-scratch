#!/usr/bin/env python3
import numpy as np
import os.path
import pickle
from .loss_functions import MeanSquaredError
from .activation_functions import Sigmoid
from .error_functions import MeanEuclideanError


class Layer:
    """Class implementation of a Neural Network Layer.
    """

    def __init__(self, dim, activation=Sigmoid(), is_output=False):
        """
        Parameters
        ----------
        dim : tuple
            the dimension of weights matrix (e.g.: ('n° previous neurons', 'n° layer's neurons') )
        activation : activation_function.ActivationFunction
            the activation function (default: Sigmoid)
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
        self.mean = 0
        self.variance = 1
        return np.random.normal(self.mean, self.variance, dim)

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
        self.A = self.activation.function(z) # sigma(net)
        self.dZ = np.atleast_2d(self.activation.derivative(z)) # partial derivative
        return self.A
    
    def backward(self, y, right_layer, loss):
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
        self.delta = loss.delta(self, right_layer, y)
        return self.delta

    def update(self, lr, left_x, alpha, lmbda):
        """Update the layer weights computing delta rule.
        
        Parameters
        ----------
        lr : float
            the learning rate
        left_x : numpy.array
            the output of previous layer and the input of current layer
        alpha : float
            the momentum parameter
        lmbda : float
            the weight decay lambda regularization parameter
        """
        x = np.atleast_2d(left_x)
        d = np.atleast_2d(self.delta)
        dw = lr * x.T.dot(d)
        self.dw_old = dw
        # add momentum
        if(self.dw_old is not None):
            momentum = alpha * self.dw_old
            dw += momentum
            self.weight -= dw
        else:
            self.weight -= dw
        # weight decay for Tikhonov regularization (L2)
        # not considering the bias
        weight_decay = 2 * lmbda * self.weight[:,1:]
        self.weight[:,1:] -= weight_decay

    
class NeuralNetwork:
    """Class implementation of an Artificial Neural Network.
    """
    
    def __init__(self, error=MeanEuclideanError(), loss=MeanSquaredError(), learn_alg='sgd'):
        """
        Parameters
        ----------
        error : error_functions.ErrorFunction
            the error function (default: MeanEuclideanError).
        loss : loss_functions.LossFunction
            the loss function (default: MeanSquaredError).
        learn_alg : string
            the learning algorithm (default: sgd).
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

    def backpropagation(self, X, y, lr=0.1, alpha=0.5, lmbda=0.01):
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
        delta = self.layers[-1].backward(y, None, self.loss)
        for l in range(len(self.layers) - 2, -1, -1):
            delta = self.layers[l].backward(delta, self.layers[l+1], self.loss)
        a = X
        # adjust weights
        for layer in self.layers:
            layer.update(lr, a, alpha, lmbda)
            a = layer.A
        return self.err.error(y, self.layers[-1].A)
    
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
        n_outputs = self.layers[-1].weight.shape[1]
        tr_errors = []
        vl_errors = []
        for k in range(epochs):
            epoch_errors = []
            for b in range(0, len(training_set), mb):
                x = np.atleast_2d(training_set[b:b+mb, :-n_outputs])
                y = np.atleast_2d(training_set[b:b+mb, -n_outputs:])
                error_mb = self.backpropagation(x, y, lr, alpha, lmbda)
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
        n_outputs = self.layers[-1].weight.shape[1]
        X = np.atleast_2d(x[:,:-n_outputs])
        y = np.atleast_2d(x[:,-n_outputs:])
        ones = np.atleast_2d(np.ones(X.shape[0]))
        a = np.concatenate((ones.T, X), axis=1)
        for l in self.layers:
            a = l.forward(a)
        return a, self.err.error(y, a)
    
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
        n_outputs = self.layers[-1].weight.shape[1]
        X = np.atleast_2d(x[:,:-n_outputs])
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
