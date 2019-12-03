# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import preprocessing

# read data from file
DATA_FILE = 'bi_cls_data.csv'
CSV_COLUMN = ['x', 'y']
data = pd.read_csv(DATA_FILE, names=CSV_COLUMN)
xs = data.x.to_numpy(np.ndarray)
ys = data.y.to_numpy(np.ndarray)

# process input data normalization
xs = preprocessing.scale(xs)

# split data to training and testing
size = len(xs)
train_size = int(size * 0.9)
X_train = xs[:train_size]
Y_train = ys[:train_size]
X_test = xs[train_size:]
Y_test = ys[train_size:]

# build neuron network
model = Sequential()
model.add(Dense(input_dim=1, activation='sigmoid', units=1))

model.compile(loss='binary_crossentropy', optimizer='sgd')

# training
print("Traing-------------------------")
for step in range(1001):
	cost = model.train_on_batch(X_train, Y_train)
	if step % 50 == 0:
		print("After %d trainings, the cost: %f" % (step, cost))

# testing
print("Testing------------------------")
cost = model.evaluate(X_test, Y_test)
print("test cost: ", cost)

# predicting
Y_pred = model.predict_classes(X_test)
print("--------------------Result------------------------")
xy_train = dict(zip(X_train, Y_train))
x_train = [k for k in sorted(xy_train.keys())]
y_train= [xy_train[k] for k in sorted(xy_train.keys())]

print("X_test: ", X_test)
print("Y_test: ", Y_test)
print("Y_pred: ", Y_pred)

# plot regression data
plt.plot(x_train, y_train)
plt.scatter(X_test, Y_pred)
plt.show()
