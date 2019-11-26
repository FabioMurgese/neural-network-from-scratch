#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import numpy as np

batch_size = 150000
lr = .3
n_epochs = 10000

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

net = nn.NeuralNetwork(dims=[2,3,1])
errors = net.fit(X, y, lr=0.1, epochs=n_epochs)

plt.plot(errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.show()

print("\ny = {:d}, y_pred = {:f}".format(0, net.predict([0, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, net.predict([0, 1])[0]))
print("y = {:d}, y_pred = {:f}".format(1, net.predict([1, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(0, net.predict([1, 1])[0]))
