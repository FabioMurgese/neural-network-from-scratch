#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import neural_network.activation_functions as activations
import neural_network.regularizers as regularizers
import neural_network.error_functions as errors
import neural_network.loss_functions as losses
import neural_network.optimizers as optimizers
import neural_network.neural_network as nn


training_set = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

# hyperparameters
lr = 0.1
epochs = 50
alpha = 0.4
lmbda = 0.001
n_hidden = 15
n_outputs = 1
mb = 4

model = nn.NeuralNetwork(
        error=errors.MeanEuclideanError(),
        loss=losses.MeanSquaredError(),
        regularizer=regularizers.L2(lmbda),
        optimizer=optimizers.SGD(lr, epochs, mb, alpha))
model.add(nn.Layer(dim=(training_set.shape[1]-n_outputs,n_hidden), activation=activations.Sigmoid()))
model.add(nn.Layer(dim=(n_hidden,1), activation=activations.Sigmoid(), is_output=True))
tr_errors, vl_errors, _, _ = model.fit(training_set, training_set)

plt.plot(tr_errors)
plt.title('Learning curve')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
