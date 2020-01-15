import numpy as np
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data
from top_k_array import get_top_k_array

def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])
    conv3 = maxpool2d(conv3, k=2)

    fc1 = tf.reshape(conv3, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,10), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}



mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

features = tf.placeholder(tf.float32, [None, 28, 28, 1], name='features')
labels = tf.placeholder(tf.float32, [None, 10], name='labels')

logits = conv_net(features, weights, biases)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

train_op = tf.train.GradientDescentOptimizer(0.05)
grads_vars = train_op.compute_gradients(cross_entropy_loss)
opt = train_op.apply_gradients(grads_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(600):
        x_batch, y_batch = mnist.train.next_batch(100)
        train_x = x_batch.reshape(-1, 28, 28, 1)
        _, gradvar, loss = sess.run([opt, grads_vars, cross_entropy_loss], {features: train_x, labels: y_batch})
        all_vars = tf.trainable_variables()

        for j, k in enumerate(zip(all_vars, gradvar)):
            if j % 2 == 0:
                var, gv = k
                g, v = gv
                if len(var.shape) == 2:
                    v = get_top_k_array(v, 0.1)
                    var.load(v, sess)

        if i % 100 == 0:
            print("The %s-th steps, loss = %f" % (i, loss))

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x = mnist.test.images.reshape(-1, 28, 28, 1)
    print("test accuracy:", accuracy.eval({features: test_x, labels: mnist.test.labels}))
