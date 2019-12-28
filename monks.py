#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import pandas as pd
import numpy as np
import datetime


# load data
dataset = pd.read_csv('datasets/monks/monks-1.train', delim_whitespace=True, header=None)
dataset_test = pd.read_csv('datasets/monks/monks-1.test', delim_whitespace=True, header=None)
training_set = dataset.iloc[:, 1:-1].values
training_set = np.hstack((training_set, np.atleast_2d(dataset.iloc[:, 0].values).T))
test_set = dataset_test.iloc[:, 1:-1].values
test_set = np.hstack((test_set, np.atleast_2d(dataset_test.iloc[:, 0].values).T))

# grid search
grid = [{"lr": 0.2, "epochs": 100, "alpha": 0.35, "lambda": 0.0002, "nhidden": 15, "mb": 10, "nfolds": 3, "activation": 'sigmoid', "loss": 'sse'},
        {"lr": 0.2, "epochs": 200, "alpha": 0.35, "lambda": 0.0003, "nhidden": 15, "mb": 10, "nfolds": 3, "activation": 'sigmoid', "loss": 'sse'},
]
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
    mb = g["mb"] # mini-batch equals to number of examples means applying SGD
    loss = g["loss"]
    n_folds = g["nfolds"]
    activation = g["activation"]
    # building the model
    model = nn.NeuralNetwork(error='mee')
    model.add(nn.Layer(dim=(training_set.shape[1]-1,n_hidden), activation=activation, loss=loss))
    model.add(nn.Layer(dim=(n_hidden,1), activation='sigmoid', is_output=True, loss=loss))
    # k-fold cross validation
    fold = 1
    for TR, VL in nn.k_fold_cross_validation(X=training_set, K=n_folds, randomise=True):
        print('Fold #{:d}'.format(fold))
        tr_errors, vl_errors = model.fit(TR, VL, lr, epochs, mb, alpha, lmbda)
        grid_tr_errors.append(tr_errors)
        grid_vl_errors.append(vl_errors)
        fold += 1
    # mean the i-th elements of the list of k-folds
    tr_errors = [0] * len(grid_tr_errors[0])
    vl_errors = [0] * len(grid_vl_errors[0])
    for lst in grid_tr_errors:
        for i, e in enumerate(lst):
            tr_errors[i] += e
    for lst in grid_vl_errors:
        for i, e in enumerate(lst):
            vl_errors[i] += e
    tr_errors = [x/n_folds for x in tr_errors]
    vl_errors = [x/n_folds for x in vl_errors]
    # plot learning curve
    plt.plot(tr_errors)
    plt.plot(vl_errors)
    plt.title('Learning curve')
    plt.xlabel('Batch')
    plt.ylabel('Error')
    plt.legend(['training', 'validation'], loc='upper right')
    desc = str(g)
    model.save(folder, desc, plt)

"""
y = test_set[:,-1]
y_pred= model.predict(test_set[:,:-1])
for i, p in enumerate(y_pred):
    print("y = {:d}, y_pred = {:f}".format(y[i], float(p)))

y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
n_equals = 0
for i, e in enumerate(y):
    if(e == y_pred[i]):
        n_equals += 1
print('Accuracy: {:f}%'.format(((n_equals/len(y))*100)))
"""