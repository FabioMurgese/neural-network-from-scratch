#!/usr/bin/env python3
import matplotlib.pyplot as plt
import neural_network as nn
import pandas as pd
import numpy as np
import datetime


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
grid = [{'lr': 0.39, 'epochs': 1000, 'alpha': 0.3, 'lambda': 0.001, 'nhidden': 2, 'mb': 15, 'nfolds': 3, 'activation': 'sigmoid', 'loss': 'mse', 'n_outputs': 1}]
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
    activation = g["activation"]
    n_outputs = g["n_outputs"]
    # building the model
    model = nn.NeuralNetwork(error='mee', loss=loss, learn_alg='sgd')
    model.add(nn.Layer(dim=(training_set.shape[1]-n_outputs,n_hidden), activation=activation))
    model.add(nn.Layer(dim=(n_hidden,1), activation='sigmoid', is_output=True))
    tr_errors, vl_errors = model.fit(training_set, test_set, lr, epochs, mb, alpha, lmbda)
    # plot learning curve
    plt.plot(tr_errors)
    #plt.plot(vl_errors)
    plt.title('Learning curve')
    plt.xlabel('Batch')
    plt.ylabel('Error')
    plt.legend(['training', 'validation'], loc='upper right')
    
    y = test_set[:,-1]
    y_pred = model.predict(test_set)
    for i, p in enumerate(y_pred):
        print("y = {:d}, y_pred = {:f}".format(int(y[i]), float(p)))
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
    n_equals = 0
    for i, e in enumerate(y):
        if(e == y_pred[i]):
            n_equals += 1
    acc = (n_equals/len(y))*100
    print('Accuracy: {:f}%'.format(acc))
    desc = str(g)
    model.save(folder, desc, plt, accuracy=acc)