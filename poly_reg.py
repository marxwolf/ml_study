import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# similar to sklearn.processing.scale()
def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X

# read data from file
DATA_FILE = 'reg_data.csv'
CSV_COLUMN = ['x', 'y']
data = pd.read_csv(DATA_FILE, names=CSV_COLUMN)
xs = data.x.to_numpy(np.ndarray)
ys = data.y.to_numpy(np.ndarray)

# process input data normalization
xs = normalize(xs)

# show the original data scatter
# plt.scatter(xs, ys)
# plt.show()

# build regression model(ploynomial)
X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

W_2 = tf.Variable(tf.random_normal([1]), name = 'weight_2')
W_3 = tf.Variable(tf.random_normal([1]), name = 'weight_3')

y_pred = tf.add(tf.multiply(X, W), b)
y_pred = tf.add(tf.multiply(tf.pow(X, 2), W_2), y_pred)
y_pred = tf.add(tf.multiply(tf.pow(X, 3), W_3), y_pred)

sample_num = xs.shape[0]
loss = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / sample_num

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict = {X:x, Y:y})
            total_loss += l
        if i % 100 == 0:
            print("Epoch {0}: {1}".format(i, total_loss / sample_num))

    W, W_2, W_3, b = sess.run([W, W_2, W_3, b])


print("W: " + str(W[0]))
print("W_2: " + str(W_2[0]))
print("W_3: " + str(W_3[0]))
print("b: " + str(b[0]))

xy_train = dict(zip(xs, ys))
x_train = [k for k in sorted(xy_train.keys())]
y_train= [xy_train[k] for k in sorted(xy_train.keys())]

plt.plot(xs, ys, 'bo', label='Real Data')
plt.plot(x_train, x_train * W + np.power(x_train, 2) * W_2 + np.power(x_train, 3) * W_3 + b, 'r', label="predicted data")
plt.legend()
plt.show()
