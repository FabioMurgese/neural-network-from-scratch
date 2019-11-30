#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

# hyperparameters
lr = .2
epochs = 10000

model = nn.NeuralNetwork()
model.add(nn.Layer(dim=(X.shape[1],3), activation='sigmoid'))
model.add(nn.Layer(dim=(3,1), activation='sigmoid', is_output=True))

errors = model.fit(X, y, lr, epochs)

plt.plot(errors)
plt.title('Learning curve')
plt.xlabel('Batch')
plt.ylabel('Error')
plt.show()

print("\ny = {:d}, y_pred = {:f}".format(0, model.predict([0, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([0, 1])[0]))
print("y = {:d}, y_pred = {:f}".format(1, model.predict([1, 0])[0]))
print("y = {:d}, y_pred = {:f}".format(0, model.predict([1, 1])[0]))
