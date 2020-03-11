#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import mnist

import neural_network.activation_functions as activations
import neural_network.regularizers as regularizers
import neural_network.error_functions as errors
import neural_network.loss_functions as losses
import neural_network.optimizers as optimizers
import neural_network.neural_network as nn


# load data
train_images = mnist.train_images()
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
test_images = mnist.test_images()
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
# one-hot_encoding targets
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
train_labels = np.atleast_2d(mnist.train_labels()).T
train_labels = encoder.fit_transform(train_labels).toarray()
test_labels = np.atleast_2d(mnist.test_labels()).T
test_labels = encoder.fit_transform(test_labels).toarray()
# build training_Set and test_set
training_set = np.hstack((train_images, train_labels))
test_set = np.hstack((test_images, test_labels))

n_outputs = 10
n_hidden = 35
epochs = 50
n_folds = 3

lr = 0.15
epochs = 25
mb = 2000
alpha = 0.2
beta = 0.9

model = nn.NeuralNetwork(
                error=errors.MeanEuclideanError(),
                loss=losses.CrossEntropy(),
                regularizer=regularizers.L2(lmbda=1e-4),
                #optimizer=optimizers.SGD(lr, epochs, mb, alpha, beta)
                #optimizer=optimizers.Adam(lr, epochs, mb)
                optimizer=optimizers.Nadam(lr, epochs, mb, alpha)
                )
model.add(nn.Layer(dim=(training_set.shape[1] - n_outputs, n_hidden), activation=activations.Sigmoid()))
model.add(nn.Layer(dim=(n_hidden, n_outputs), activation=activations.Softmax(), is_output=True))

# k-fold cross validation
kf_tr_errors = []
kf_vl_errors = []
n = 1
for TR, VL in nn.k_fold_cross_validation(X=training_set, K=n_folds, shuffle=True):
    print('Fold #', n)
    tr_errors, vl_errors, _, _ = model.fit(TR, VL, verbose=True)
    kf_tr_errors.append(tr_errors)
    kf_vl_errors.append(vl_errors)
    n += 1

tr_errors = [0] * epochs
vl_errors = [0] * epochs
for lst in kf_tr_errors:
    for i, e in enumerate(lst):
        tr_errors[i] += e
    for lst in kf_vl_errors:
        for i, e in enumerate(lst):
            vl_errors[i] += e
    tr_errors = [x/n_folds for x in tr_errors]
    vl_errors = [x/n_folds for x in vl_errors]

learning_img, plt1 = plt.subplots()
plt1.plot(tr_errors)
plt1.plot(vl_errors)
plt1.set_title("Learning curve")
plt1.set_xlabel("Epochs")
plt1.set_ylabel("Error")
plt1.legend(['train', 'validation'], loc='upper right')
plt.close()

now = datetime.datetime.now()
folder = "{0}".format(now.strftime('%Y%m%d_%H%M%S'))
model.save(folder, '', learning_img)
