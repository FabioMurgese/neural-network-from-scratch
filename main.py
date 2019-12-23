#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import numpy as np

training_set = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

# hyperparameters
lr = 0.1
epochs = 1000
alpha = 0.4
lmbda = 0.001
n_hidden = 15
mb = 4 # mini-batch equals to number of examples means applying SGD
loss = 'sse'

model = nn.NeuralNetwork()
model.add(nn.Layer(dim=(training_set.shape[1]-1,n_hidden), activation='sigmoid', loss=loss))
model.add(nn.Layer(dim=(n_hidden,1), activation='sigmoid', is_output=True, loss=loss))

errors = model.fit(training_set, lr, epochs, mb, alpha, lmbda)

plt.plot(errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.show()

"""
filename = model.save()
model = nn.NeuralNetwork()
model = model.load(filename)
"""
