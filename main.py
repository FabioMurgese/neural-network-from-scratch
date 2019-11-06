#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import random


batch_size = 15000
lr = .3
net = nn.NeuralNetwork(sizes=[3,1])

training_set = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

errors = []
for i in range(batch_size):
    mse = net.backprop(training_set, lr=lr)
    errors.append(mse)
    print("batch: {:d}/{:d}, mse: {:f}".format(i+1, batch_size, mse))
    random.shuffle(training_set)

plt.plot(errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.show()

print("\ny = {:d}, y_pred = {:f}".format(0, net.predict([0, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, net.predict([0, 1])[0]))
print("y = {:d}, y_pred = {:f}".format(1, net.predict([1, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, net.predict([1, 1])[0]))
