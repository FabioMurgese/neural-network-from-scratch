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
lr = .1
epochs = 10000

model = nn.NeuralNetwork()
model.add(nn.Layer(dim=(training_set.shape[1]-1,4), activation='sigmoid'))
model.add(nn.Layer(dim=(4,1), activation='sigmoid', is_output=True))

errors = model.fit(training_set, lr, epochs)

plt.plot(errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.show()

print("\ny = {:d}, y_pred = {:f}".format(0, model.predict([0, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([0, 1])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([1, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(0, model.predict([1, 1])[0]))

filename = model.save()
model = nn.NeuralNetwork()
model = model.load(filename)
print("\ny = {:d}, y_pred = {:f}".format(0, model.predict([0, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([0, 1])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([1, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(0, model.predict([1, 1])[0]))
