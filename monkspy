#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
dataset = pd.read_csv('datasets/monks/monks-3.train', delim_whitespace=True, header=None)
dataset_test = pd.read_csv('datasets/monks/monks-3.test', delim_whitespace=True, header=None)
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
grid = [{'lr': 0.23, 'epochs': 400, 'alpha': 0.2, 'lambda': 1e-05, 'nhidden': 4, 'mb': 20, 'nfolds': 4, 'activation': activations.Sigmoid(), 'loss': losses.MeanSquaredError(), 'n_outputs': 1}]
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
    model = nn.Sequential(
            error=errors.MeanSquaredError(),
            loss=losses.MeanSquaredError(),
            #regularizer=None,
            regularizer=regularizers.L2(lmbda),
            optimizer=optimizers.SGD(lr, epochs, mb, alpha)
            #optimizer=optimizers.Adam(alpha=lr, epochs=epochs)
    )
    model.add(nn.Dense(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activation))
    model.add(nn.Dense(dim=(n_hidden, 1), activation=activations.Sigmoid(), is_output=True))
    tr_errors, vl_errors, tr_accuracy, vl_accuracy = model.fit(training_set, test_set, True)

    _, MSE_test_set = model.validate(test_set)

    # plot learning curve
    learning_img, plt1 = plt.subplots()
    plt1.plot(tr_errors)
    plt1.plot(vl_errors)
    plt1.set_title("Learning curve")
    plt1.set_xlabel("Epochs")
    plt1.set_ylabel("Error")
    plt1.legend(['train', 'validation'], loc='upper right')
    learning_img.show()
    plt.close(learning_img)

    # plot accuracy curve
    accuracy_img, plt2 = plt.subplots()
    plt2.plot(tr_accuracy)
    plt2.plot(vl_accuracy)
    plt2.set_title("Accuracy")
    plt2.set_xlabel("Epochs")
    plt2.set_ylabel("% Accuracy")
    plt2.legend(['train', 'validation'], loc='lower right')
    #accuracy_img.show()
    plt.close(accuracy_img)

    y = test_set[:, -1]
    y_pred = model.predict(test_set[:, :-1])
    """
    for i, p in enumerate(y_pred):
        print("y = {:d}, y_pred = {:f}".format(int(y[i]), float(p)))"""
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
    n_equals = 0
    for i, e in enumerate(y):
        if(e == y_pred[i]):
            n_equals += 1
    acc = (n_equals/len(y))*100
    print('Accuracy: {:f}%'.format(acc))
    g["activation"] = type(activation).__name__
    g["loss"] = type(model.loss).__name__
    desc = str(g) + '\nAccuracy: ' + str(acc) + "\nMSE training set: {0}".format(tr_errors[-1]) \
                        + "\nMSE test set: {0}".format(MSE_test_set)
    model.save(folder, desc, learning_img, accuracy_img)
