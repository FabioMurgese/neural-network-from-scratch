#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import pandas as pd
import numpy as np


# load data
dataset = pd.read_csv('datasets/monks/monks-1.train', delim_whitespace=True, header=None)
dataset_test = pd.read_csv('datasets/monks/monks-1.test', delim_whitespace=True, header=None)
training_set = dataset.iloc[:, 1:-1].values
training_set = np.hstack((training_set, np.atleast_2d(dataset.iloc[:, 0].values).T))
test_set = dataset_test.iloc[:, 1:-1].values
test_set = np.hstack((test_set, np.atleast_2d(dataset_test.iloc[:, 0].values).T))

# grid search
grid = [
        {"lr": 0.2, "epochs": 300, "alpha": 0.3, "lambda": 0.0001, "nhidden": 10, "mb": 5, "loss": 'sse'}]
grid_tr_errors = []
grid_vl_errors = []
k = 3
for g in grid:
    # hyperparameters
    lr = g["lr"]
    epochs = g["epochs"]
    alpha = g["alpha"]
    lmbda = g["lambda"]
    n_hidden = g["nhidden"]
    mb = g["mb"] # mini-batch equals to number of examples means applying SGD
    loss = g["loss"]
    # building the model
    model = nn.NeuralNetwork(error='mee')
    model.add(nn.Layer(dim=(training_set.shape[1]-1,n_hidden), activation='sigmoid', loss=loss))
    model.add(nn.Layer(dim=(n_hidden,1), activation='sigmoid', is_output=True, loss=loss))
    # k-fold cross validation
    fold = 1
    for TR, VL in nn.k_fold_cross_validation(X=training_set, K=k, randomise=True):
        print('Fold #{:d}'.format(fold))
        tr_errors, vl_errors = model.fit(TR, VL, lr, epochs, mb, alpha, lmbda)
        grid_tr_errors.append(tr_errors)
        grid_vl_errors.append(vl_errors)
        fold += 1

# mean the i-th elements of the list of k-folds
tr_errors = [0] * len(grid_tr_errors[0])
for lst in grid_tr_errors:
    for i, e in enumerate(lst):
        tr_errors[i] += e
tr_errors = [x/k for x in tr_errors]
vl_errors = [0] * len(grid_vl_errors[0])
for lst in grid_vl_errors:
    for i, e in enumerate(lst):
        vl_errors[i] += e
vl_errors = [x/k for x in vl_errors]

# plot learning curve
plt.plot(tr_errors)
plt.plot(vl_errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

y = test_set[:,-1]
y_pred, _ = model.predict(test_set)
"""
for i, p in enumerate(y_pred):
    print("y = {:d}, y_pred = {:f}".format(y[i], float(p)))
"""
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
n_equals = 0
for i, e in enumerate(y):
    if(e == y_pred[i]):
        n_equals += 1
print('Accuracy: {:f}%'.format(((n_equals/len(y))*100)))
