#!/usr/bin/env python3
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
        [0.05, 0.04, 0.03, 0.02, 0.01], # learning rates
        [500], # epochs
        [0.9], # momentum alphas
        [0.8], # momentum betas (moving average)
        [1e-08, 1e-07, 1e-06, 1e-05], # lambdas
        [20], # hidden units
        [100], # mini-batches
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
        model = nn.NeuralNetwork(
                error=errors.MeanEuclideanError(),
                loss=losses.MeanSquaredError(),
                regularizer=regularizers.L2(lmbda),
                optimizer=optimizers.SGD(lr, epochs, mb, alpha, beta))
        model.add(nn.Layer(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activation))
        model.add(nn.Layer(dim=(n_hidden, n_hidden), activation=activation))
        model.add(nn.Layer(dim=(n_hidden, n_outputs), activation=activations.Linear(), is_output=True))
        
        start_time = datetime.datetime.now()
        # k-fold cross validation
        for TR, VL in nn.k_fold_cross_validation(X=training_set, K=n_folds, shuffle=True):
            tr_errors, vl_errors, _, _ = model.fit(TR, VL)
            grid_tr_errors.append(tr_errors)
            grid_vl_errors.append(vl_errors)
        end_time = datetime.datetime.now()
        time = end_time - start_time
        
        _, MEE_inner_test_set = model.validate(test_set)
        variance = np.var(grid_tr_errors)
        
        # mean the i-th elements of the list of k-folds
        min_len_tr = min(map(len, grid_tr_errors))
        min_len_vl = min(map(len, grid_vl_errors))
        tr_errors = [0] * min_len_tr
        vl_errors = [0] * min_len_vl
        for lst in grid_tr_errors:
            for i in range(len(tr_errors)):
                tr_errors[i] += lst[i]
        for lst in grid_vl_errors:
            for i in range(len(vl_errors)):
                vl_errors[i] += lst[i]
        tr_errors = [x/n_folds for x in tr_errors]
        vl_errors = [x/n_folds for x in vl_errors]
    
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

# model assessment
#model.predict(test_set[:, :-2], save_csv=True)

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
