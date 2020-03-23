#!/usr/bin/env python3
import numpy as np
import os.path
import pickle

from .error_functions import MeanEuclideanError
from .loss_functions import MeanSquaredError
from .activation_functions import Sigmoid
from .regularizers import L2
from .optimizers import SGD


class Layer(object):
    """Layer interface.
    """
    
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
        raise NotImplementedError
        
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
        raise NotImplementedError


class NeuralNetwork(object):
    """Neural Network abstract class.
    """
    
    def load(self, filename):
        """Load NeuralNetwork object from file.
        
        Returns
        -------
        the NeuralNetwork object
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def save(self, folder, description='', fig1=None, fig2=None):
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
        if(fig1 is not None):
            fig1.savefig(os.path.join(directory, 'learning_curve.png'))
        if (fig2 is not None):
            fig2.savefig(os.path.join(directory, 'accuracy_curve.png'))
        desc_filename = os.path.join(directory, 'description')
        with open(desc_filename, 'w') as file:
            file.write(description)
        return filename
    
    def fit(self, training_set, validation_set, compute_accuracy=False, verbose=False):
        """Computes the default optimization learning algorithm.
        
        Parameters
        ----------
        training_set : numpy.array
            the training set
        validation_set : numpy.array
            the validation set
        
        Returns
        -------
        the training errors, the validation errors
        """
        raise NotImplementedError
        
    def feedforward(self, x):
        """Feedforward the inputs throw the network.
        
        Parameters
        ----------
        x : numpy.array
            the inputs data
        """
        raise NotImplementedError
        
    def compute_deltas(self, y):
        """Computes the deltas of the network.
        
        Parameters
        ----------
        y : numpy.array
            the targets
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def predict(self, x, save_csv=False):
        """Computes the predicted output of the network.
        
        Parameters
        ----------
        x : numpy.array
            the inputs
        save_csv : bool
            the flag that shows if a .csv file has to be saved
        Returns
        -------
        the predicted output
        """
        raise NotImplementedError
        
        

class Dense(Layer):
    """Class implementation of a Dense Neural Network Layer.
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
        self.input = None
        self.output = None
        self.dw_old = np.zeros_like(self.weight)
        self.is_output_layer = is_output

    def __normal_weights(self, dim):
        """Initialize a matrix with normal distributed rows.
        """
        self.mean = 0
        self.standard_deviation = 1
        return np.random.normal(self.mean, self.standard_deviation, dim)

    def forward(self, x):
        self.input = x
        net = x.dot(self.weight) # the net of the layer
        self.output = self.activation.function(net) # sigma(net)
        self.dz = np.atleast_2d(self.activation.derivative(net)) # partial derivative
        return self.output
    
    def backward(self, y, right_layer, loss):
        self.delta = loss.delta(self, right_layer, y)
        return self.delta

    
class Sequential(NeuralNetwork):
    """Class implementation of a fully-connected Artificial Neural Network.
    """
    
    def __init__(self, error=MeanEuclideanError(), loss=MeanSquaredError(), regularizer=L2(), optimizer=SGD()):
        """
        Parameters
        ----------
        error : error_functions.ErrorFunction
            the error function (default: MeanEuclideanError)
        loss : loss_functions.LossFunction
            the loss function (default: MeanSquaredError)
        regularizer : regularizers.Regularizer
            the regularizer (default: L2)
        optimizer : optimizers.Optimizer
            the optimizer (default: SGD)
        """
        self.layers = []
        self.error = error
        self.loss = loss
        self.regularizer = regularizer
        self.optimizer = optimizer
    
    def n_outputs(self):
        """
        Returns
        -------
        the number of outputs of the network
        """
        return self.layers[-1].weight.shape[1]
        
    def add(self, layer):
        """Add a new layer to the network.
        
        Parameters
        ----------
        layer : neural_network.Layer
            the new layer
        """
        self.layers.append(layer)
    
    def save(self, folder, description='', fig1=None, fig2=None):
        return super().save(folder, description, fig1, fig2)
    
    def load(self, filename):
        return super().load(filename)
    
    def fit(self, training_set, validation_set, compute_accuracy=False, verbose=False):
        tr_errors, vl_errors, tr_accuracy, vl_accuracy = self.optimizer.train(training_set, validation_set, self, compute_accuracy, verbose)
        return tr_errors, vl_errors, tr_accuracy, vl_accuracy
    
    def feedforward(self, x):
        ones = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((ones.T, x), axis=1)
        for layer in self.layers:
            x = layer.forward(x)
    
    def compute_deltas(self, y):
        delta = self.layers[-1].backward(y, None, self.loss)
        for l in range(len(self.layers) - 2, -1, -1):
            delta = self.layers[l].backward(delta, self.layers[l+1], self.loss)
    
    def validate(self, x):
        X = np.atleast_2d(x[:, :-self.n_outputs()])
        y = np.atleast_2d(x[:, -self.n_outputs():])
        ones = np.atleast_2d(np.ones(X.shape[0]))
        a = np.concatenate((ones.T, X), axis=1)
        for l in self.layers:
            a = l.forward(a)
        return a, self.error.error(y, a)
    
    def predict(self, x, save_csv=False):
        ones = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((ones.T, x), axis=1)
        for l in self.layers:
            x = l.forward(x)
        if save_csv:
            np.savetxt('predictions.csv', x, delimiter=',')
        return x

    def compute_accuracy(self, targets, predictions):
        predictions = [1 if x >= 0.5 else 0 for x in predictions]
        n_equals = 0
        for i, e in enumerate(targets):
            if (e == predictions[i]):
                n_equals += 1
        accuracy = (n_equals / len(targets)) * 100
        return accuracy


def k_fold_cross_validation(X, K, shuffle=False):
    """Perform k-fold cross validation splitting dataset
    in trainng set and validation set.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K, shuffle=shuffle)
    kf.get_n_splits(X)
    for tr_index, val_index in kf.split(X):
        training, validation = X[tr_index], X[val_index]
        yield training, validation

def get_grid_search(*args):
    """Get the cartesian product for a grid search 
    given a list of lists of parameters.
    
    Parameters
    ----------
    args : *args
        [[learning rates      ], 
         [epochs              ],
         [alphas              ],
         [betas               ],
         [lambdas             ],
         [hidden units        ],
         [minibatches         ],
         [number of folds     ],
         [activation functions]]
    
    Returns
    -------
    the grid search
    """
    import itertools
    grid = []
    for e in itertools.product(*args):
        grid.append({'lr': e[0],
                     'epochs': e[1],
                     'alpha': e[2],
                     'beta': e[3],
                     'lambda': e[4],
                     'nhidden': e[5],
                     'mb': e[6],
                     'nfolds': e[7],
                     'activation': e[8]})
    return grid
