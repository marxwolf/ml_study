# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt 
from sklearn import preprocessing

# read data from file
DATA_FILE = 'cls_data.csv'
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

# one-hot encoding of classification
one_hot_Y_train = to_categorical(Y_train)
one_hot_Y_test = to_categorical(Y_test)

# build neuron network
model = Sequential()
model.add(Dense(input_dim=1, activation='relu', units=64))
model.add(Dense(activation='linear', units=64))
model.add(Dense(activation='softmax', output_dim=4))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
print("\nTraining-------------------------")
cost = model.fit(X_train, one_hot_Y_train, epochs=100, verbose=1)

# testing
print("\nTesting-------------------------")
cost = model.evaluate(X_test, one_hot_Y_test)
print("test cost: ", cost)

# predicting
Y_pred = model.predict(X_test)
print("X_test: ", X_test)
print("Y_Pred: ", Y_pred)
print("Y_pred after decode: ", np.argmax(Y_pred, axis=1))
