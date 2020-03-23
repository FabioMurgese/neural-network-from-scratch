#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
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
dataset = pd.read_csv('datasets/cup/ML-CUP19-TR.csv', header=None)
dataset_test = pd.read_csv('datasets/cup/ML-CUP19-TS.csv', header=None)

training_set = dataset.iloc[:1300, :].values
test_set = dataset.iloc[1300:, :].values
blind_test_set = dataset_test.iloc[:, :].values
dataset = dataset.iloc[:, :].values

# model selection
# grid search
grid = nn.get_grid_search(
        [0.001, 0.002, 0.01, 0.02, 0.1, 0.2], # learning rates
        [50, 100, 500], # epochs
        [0.9, 0.7, 0.4, 0.2, 0.1, 0.05], # momentum alphas
        [0.01], # momentum betas (moving average)
        [1e-08, 1e-07, 1e-06], # lambdas
        [20], # hidden units
        [50], # mini-batches
        [5], # number of folds
        [activations.Sigmoid()] # activation functions
)

now = datetime.datetime.now()
with tqdm(total=int(len(grid)), position=0, leave=True) as progress_bar:
    for i, g in enumerate(grid):
        folder = "{0}_{1}".format(now.strftime('%Y%m%d_%H%M%S'), i+1)
        grid_tr_errors = []
        grid_vl_errors = []
        n_outputs = 2
        
        # hyperparameters
        lr = g["lr"]
        epochs = g["epochs"]
        alpha = g["alpha"]
        beta = g["beta"]
        n_hidden = g["nhidden"]
        mb = g["mb"]
        n_folds = g["nfolds"]
        activation = g["activation"]
        lmbda = g["lambda"]
        
        # building the model
        model = nn.Sequential(
                error=errors.MeanEuclideanError(),
                loss=losses.MeanSquaredError(),
                regularizer=regularizers.L2(lmbda),
                #optimizer=optimizers.SGD(lr, epochs, mb, alpha, beta)
                optimizer=optimizers.Nadam(lr, epochs, mb, alpha)
                #optimizer=optimizers.Adam(lr, epochs, mb, lr_decay=True)
                )
        model.add(nn.Dense(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activation))
        model.add(nn.Dense(dim=(n_hidden, n_hidden), activation=activation))
        model.add(nn.Dense(dim=(n_hidden, n_hidden), activation=activation))
        model.add(nn.Dense(dim=(n_hidden, n_outputs), activation=activations.Linear(), is_output=True))
        
        start_time = datetime.datetime.now()
        # k-fold cross validation
        for TR, VL in nn.k_fold_cross_validation(X=training_set, K=n_folds, shuffle=True):
            tr_errors, vl_errors, _, _ = model.fit(TR, VL)
            grid_tr_errors.append(tr_errors)
            grid_vl_errors.append(vl_errors)
        end_time = datetime.datetime.now()
        time = end_time - start_time
        
        # mean the i-th elements of the list of k-folds
        tr_errors = [0] * epochs
        vl_errors = [0] * epochs
        for lst in grid_tr_errors:
            for i, e in enumerate(lst):
                tr_errors[i] += e
        for lst in grid_vl_errors:
            for i, e in enumerate(lst):
                vl_errors[i] += e
        tr_errors = [x/n_folds for x in tr_errors]
        vl_errors = [x/n_folds for x in vl_errors]
        
        _, MEE_inner_test_set = model.validate(test_set)
        variance = np.var(grid_tr_errors)
    
        # plot learning curve
        learning_img, plt1 = plt.subplots()
        plt1.plot(tr_errors)
        plt1.plot(vl_errors)
        plt1.set_title("Learning curve")
        plt1.set_xlabel("Epochs")
        plt1.set_ylabel("Error")
        plt1.legend(['train', 'validation'], loc='upper right')
        plt.close()
        
        g["optimizer"] = type(model.optimizer).__name__
        g["regularizer"] = type(model.regularizer).__name__
        g["activation"] = type(activation).__name__
        g["loss"] = type(model.loss).__name__
        desc = str(g) \
                + "\nMEE TR: {0}".format(tr_errors[-1]) \
                + "\nMEE VL: {0}".format(vl_errors[-1]) \
                + "\nMEE TS (inner): {0}".format(MEE_inner_test_set) \
                + "\nVariance TR: {0}".format(variance) \
                + "\nTrained in {0} seconds".format(str(time.total_seconds()))
        model.save(folder, desc, learning_img)
        progress_bar.update(1)

# extract and order the models w.r.t MEE VL
import os
runs_dir = 'runs/'
models_mee = []
for folder in os.listdir(runs_dir):
    file = open(os.path.join(runs_dir, folder, 'description'))
    for i, line in enumerate(file):
        if i == 2:
            mee = float(line.split(': ')[1])
            models_mee.append({'name': folder, 'mee': mee})
models_mee = sorted(models_mee, key=lambda i: i["mee"])
print(models_mee)

"""
# model assessment
n_outputs = 2
n_hidden = 20
activation = activations.Sigmoid()
model = nn.Sequential(
            error=errors.MeanEuclideanError(),
            loss=losses.MeanSquaredError(),
            regularizer=regularizers.L2(lmbda=1e-07),
            optimizer=optimizers.Adam(lr=0.8, epochs=800, mb=50, lr_decay=True)
            )
model.add(nn.Dense(dim=(dataset.shape[1] - n_outputs, n_hidden), activation=activation))
model.add(nn.Dense(dim=(n_hidden, n_hidden), activation=activation))
model.add(nn.Dense(dim=(n_hidden, n_hidden), activation=activation))
model.add(nn.Dense(dim=(n_hidden, n_outputs), activation=activations.Linear(), is_output=True))
tr_errors, _, _, _ = model.fit(dataset, dataset, verbose=True)

# plot learning curve
learning_img, plt1 = plt.subplots()
plt1.plot(tr_errors)
plt1.set_title("Model assessment")
plt1.set_xlabel("Epochs")
plt1.set_ylabel("Error")
plt1.legend(['dataset'], loc='upper right')
learning_img.show()

_, ts_error = model.validate(test_set)
print('Training error:', tr_errors[-1])
print('Test error:', ts_error)
model.save('final_model')

#model = nn.NeuralNetwork().load('/home/anto/Programming/neural-network-from-scratch/models/cup_adam/20200323_155217_14/final_model.pkl')
#model.predict(blind_test_set, save_csv=True)
"""