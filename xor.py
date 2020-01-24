#!/usr/bin/env python3
import matplotlib.pyplot as plt
from neural_network import neural_network as nn
import neural_network.loss_functions as losses
import neural_network.activation_functions as activations
import neural_network.error_functions as errors
import numpy as np

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
mb = 4 # mini-batch equals to number of examples means applying SGD
loss = losses.MeanSquaredError()

model = nn.NeuralNetwork(error=errors.MeanEuclideanError(), loss=loss, learn_alg='sgd')
model.add(nn.Layer(dim=(training_set.shape[1]-n_outputs,n_hidden), activation=activations.Sigmoid()))
model.add(nn.Layer(dim=(n_hidden,1), activation=activations.Sigmoid(), is_output=True))
tr_errors, vl_errors = model.fit(training_set, training_set, lr, epochs, mb, alpha, lmbda)

plt.plot(tr_errors)
plt.title('Learning curve')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
