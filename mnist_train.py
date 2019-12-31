import numpy as np
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

def get_top_k(t, k):
	if len(t.shape) == 1:
		values, _ = tf.nn.top_k(t, k)
	elif len(t.shape) == 2:
		flatten_t = tf.reshape(t, [-1])
		values, _ = tf.nn.top_k(flatten_t, k)
	
	threshold = values[-1]

	mask = tf.greater_equal(t, threshold)
	zeros = tf.zeros_like(t)
	top_k = tf.where(mask, t, zeros)

	return top_k


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.zeros([784, 10]))

b1 = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W1) + b1)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_op = tf.train.GradientDescentOptimizer(0.01)
w = tf.trainable_variables()[0]
grads_vars = train_op.compute_gradients(cross_entropy, var_list=[w])
# residual = tf.zeros([784, 10])

for i, (g, v) in enumerate(grads_vars):
    # import pdb; pdb.set_trace()
    # if len(g.shape) == 2:
    #     g_size = 1
    #     g_shape = g.get_shape().as_list()
    #     for j in g_shape:
    #         g_size *= j
    #     top_k_size = int(g_size * 0.1)
    #     grads_vars[i] = (get_top_k(g, top_k_size), v)
        # g = tf.add(g, residual)
        # g_top_k = get_top_k(g, top_k_size)
        # residual = tf.subtract(g, g_top_k)
        # grads_vars[i] = (g_top_k, v)
    if len(v.shape) == 2:
        v_size = 1
        v_shape = v.get_shape().as_list()
        for j in v_shape:
            v_size *= j
        top_k_size = int(v_size * 0.1)
        grads_vars[i] = (g, get_top_k(v, top_k_size))

opt = train_op.apply_gradients(grads_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        _, gradvar, loss = sess.run([opt, grads_vars, cross_entropy], {x: x_batch, y_: y_batch})
        # import pdb; pdb.set_trace()
        
        if i % 100 == 0:
            print("The %s-th steps, loss = %f" % (i, loss))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))