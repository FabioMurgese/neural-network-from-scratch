#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
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

# model selection
# grid search
grid = nn.get_grid_search(
        [0.1, 0.01, 0.001, 0.2, 0.02, 0.002, 0.3, 0.03, 0.003, 0.4, 0.04, 0.004], # learning rates
        [600, 1200, 2500], # epochs
        [0.01, 0.1, 0.2, 0.3, 0.4], # alphas
        [0.001, 0.0001, 0.00001, 0.000001, 0.0000001], # lambdas
        [7, 20, 50, 100], # hidden units
        [50, 120, 250, 300, 350], # mini-batches
        [5, 8, 10], # number of folds
        [activations.Sigmoid(), activations.ReLu(), activations.Tanh()] # activation functions
)

now = datetime.datetime.now()
for i, g in enumerate(grid):
    folder = "{0}_{1}".format(now.strftime('%Y%m%d_%H%M%S'), i+1)
    grid_tr_errors = []
    grid_vl_errors = []
    # hyperparameters
    lr = g["lr"]
    epochs = g["epochs"]
    alpha = g["alpha"]
    n_hidden = g["nhidden"]
    mb = g["mb"]
    loss = g["loss"]
    n_folds = g["nfolds"]
    activation = g["activation"]
    n_outputs = g["n_outputs"]
    lmbda = g["lambda"]
    # building the model
    model = nn.NeuralNetwork(
            error=errors.MeanEuclideanError(),
            loss=loss,
            regularizer=regularizers.L2(lmbda),
            optimizer=optimizers.SGD(lr, epochs, mb, alpha))
    model.add(nn.Layer(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activation))
    model.add(nn.Layer(dim=(n_hidden, n_hidden), activation=activation))
    model.add(nn.Layer(dim=(n_hidden, 2), activation=activations.Linear(), is_output=True))

    start_time = datetime.datetime.now()
    # k-fold cross validation
    fold = 1
    for TR, VL in nn.k_fold_cross_validation(X=training_set, K=n_folds, shuffle=True):
        print('Fold #{:d}'.format(fold))
        tr_errors, vl_errors, _, _ = model.fit(TR, VL)
        grid_tr_errors.append(tr_errors)
        grid_vl_errors.append(vl_errors)
        fold += 1
    end_time = datetime.datetime.now()
    time = end_time - start_time
    print("Trained in {0} seconds".format(str(time.total_seconds())))

    _, MEE_inner_test_set = model.validate(test_set)
    print("Mean Euclidean Error inner test_set: {0}".format(MEE_inner_test_set))

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
    learning_img, plt1 = plt.subplots()
    plt1.plot(tr_errors)
    plt1.plot(vl_errors)
    plt1.set_title("Learning curve")
    plt1.set_xlabel("Epochs")
    plt1.set_ylabel("Error")
    plt1.legend(['train', 'validation'], loc='upper right')
    learning_img.show()
    plt.close(learning_img)

    g["activation"] = type(activation).__name__
    g["loss"] = type(loss).__name__
    desc = str(g) + "\nMean Euclidean Error training set: {0}".format(tr_errors[-1]) \
                        + "\nMean Euclidean Error validation set: {0}".format(vl_errors[-1]) \
                        + "\nMean Euclidean Error inner test set: {0}".format(MEE_inner_test_set)
    model.save(folder, desc, learning_img)
    model.predict(test_set[:, :-2], save_csv=True)

# model assessment

#model = nn.NeuralNetwork().load('models/cup/20200124_125548_1/20200124_192433_1.pkl')
#model.predict(test_set, save_csv=True)
