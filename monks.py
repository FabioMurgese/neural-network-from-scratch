#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import neural_network.activation_functions as activations
import neural_network.regularizers as regularizers
import neural_network.error_functions as errors
import neural_network.loss_functions as losses
import neural_network.optimizers as optimizers
import neural_network.neural_network as nn


# load data
dataset = pd.read_csv('datasets/monks/monks-1.train', delim_whitespace=True, header=None)
dataset_test = pd.read_csv('datasets/monks/monks-1.test', delim_whitespace=True, header=None)
training_set = dataset.iloc[:, 1:-1].values
test_set = dataset_test.iloc[:, 1:-1].values
# One-Hot Encoding training set
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
training_set = encoder.fit_transform(training_set).toarray()
training_set = np.hstack((training_set, np.atleast_2d(dataset.iloc[:, 0].values).T))
# One-Hot Encoding test set
test_set = encoder.fit_transform(test_set).toarray()
test_set = np.hstack((test_set, np.atleast_2d(dataset_test.iloc[:, 0].values).T))

# grid search
grid = [{'lr': 0.4, 'epochs': 1000, 'alpha': 0.3, 'lambda': 1e-4, 'nhidden': 3, 'mb': 15, 'nfolds': 4, 'activation': activations.Sigmoid(), 'loss': losses.MeanSquaredError(), 'n_outputs': 1}]
now = datetime.datetime.now()
for i, g in enumerate(grid):
    folder = "{0}_{1}".format(now.strftime('%Y%m%d_%H%M%S'), i+1)
    grid_tr_errors = []
    grid_vl_errors = []
    # hyperparameters
    lr = g["lr"]
    epochs = g["epochs"]
    alpha = g["alpha"]
    lmbda = g["lambda"]
    n_hidden = g["nhidden"]
    mb = g["mb"]
    activation = g["activation"]
    n_outputs = g["n_outputs"]
    # building the model
    model = nn.NeuralNetwork(
            error=errors.MeanEuclideanError(),
            loss=losses.MeanSquaredError(),
            regularizer=regularizers.L2(lmbda),
            optimizer=optimizers.SGD(lr, epochs, mb, alpha)
            #optimizer=optimizers.Adam(alpha=lr, epochs=epochs)
    )
    model.add(nn.Layer(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activation))
    model.add(nn.Layer(dim=(n_hidden, 1), activation=activations.Sigmoid(), is_output=True))
    tr_errors, vl_errors, tr_accuracy, vl_accuracy = model.fit(training_set, test_set, True)

    # plot learning curve
    plt.plot(tr_errors)
    plt.plot(vl_errors)
    plt.title('Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.close()

    # plot accuracy curve
    plt.plot(tr_accuracy)
    plt.plot(vl_accuracy)
    plt.title('Accuracy curve')
    plt.xlabel('Epochs')
    plt.ylabel('% Accuracy')
    plt.legend(['train', 'validation'], loc='bottom right')
    plt.show()
    plt.close()

    y = test_set[:,-1]
    y_pred = model.predict(test_set[:,:-1])
    for i, p in enumerate(y_pred):
        print("y = {:d}, y_pred = {:f}".format(int(y[i]), float(p)))
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
    n_equals = 0
    for i, e in enumerate(y):
        if(e == y_pred[i]):
            n_equals += 1
    acc = (n_equals/len(y))*100
    print('Accuracy: {:f}%'.format(acc))
    g["activation"] = type(activation).__name__
    g["loss"] = type(model.loss).__name__
    desc = str(g)
    model.save(folder, desc, plt, accuracy=acc)
