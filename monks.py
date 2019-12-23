#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antodima
"""
import matplotlib.pyplot as plt
import neural_network as nn
import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets/monks/monks-1.train', delim_whitespace=True, header=-1)
training_set = dataset.iloc[:, 1:-1].values
training_set = np.hstack((training_set, np.atleast_2d(dataset.iloc[:, 0].values).T))

dataset_test = pd.read_csv('datasets/monks/monks-1.test', delim_whitespace=True, header=-1)
test_set = dataset_test.iloc[:, 1:-1].values
test_set = np.hstack((test_set, np.atleast_2d(dataset_test.iloc[:, 0].values).T))


# hyperparameters
lr = 0.1
epochs = 50
alpha = 0.4
lmbda = 0.001
n_hidden = 15
mb = 5 # mini-batch equals to number of examples means applying SGD
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

y = test_set[:,-1]
y_pred = model.predict(test_set[:,:-1])
for i, p in enumerate(y_pred):
    print("y = {:d}, y_pred = {:f}".format(y[i], float(p)))

y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
n_equals = 0
for i, e in enumerate(y):
    if(e == y_pred[i]):
        n_equals += 1
print('Accuracy: ', (n_equals/len(y)))
