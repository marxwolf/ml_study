import numpy as np
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data
from top_k_array import get_top_k_array

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

features = tf.placeholder(tf.float32, [None, 784], name='features')
labels = tf.placeholder(tf.float32, [None, 10], name='labels')

dense1 = tf.layers.dense(inputs=features, units=256)
dense2 = tf.layers.dense(inputs=dense1, units=128)
dense3 = tf.layers.dense(inputs=dense2, units=64)
dense4 = tf.layers.dense(inputs=dense3, units=32)
logits = tf.layers.dense(inputs=dense4, units=10)

classes = tf.argmax(logits)
probabilities = tf.nn.softmax(logits)

# cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

train_op = tf.train.GradientDescentOptimizer(0.005)
w = tf.trainable_variables()[0]
grads_vars = train_op.compute_gradients(cross_entropy_loss, var_list=[w])


# for i, (g, v) in enumerate(grads_vars):
    # import pdb; pdb.set_trace()
    # if len(g.shape) == 2:
    #     g_size = 1
    #     g_shape = g.get_shape().as_list()
    #     for j in g_shape:
    #         g_size *= j
    #     top_k_size = int(g_size * 1)
    #     g_top_k = get_top_k(g, top_k_size)
    #     grads_vars[i] = (g_top_k, v)
    # if len(v.shape) == 2:
    #     v_size = 1
    #     v_shape = v.get_shape().as_list()
    #     for j in v_shape:
    #         v_size *= j
    #     top_k_size = int(v_size * 0.001)
    #     grads_vars[i] = (g, get_top_k(v, top_k_size))

opt = train_op.apply_gradients(grads_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_batch, y_batch = mnist.train.next_batch(128)
        _, gradvar, loss = sess.run([opt, grads_vars, cross_entropy_loss], {features: x_batch, labels: y_batch})
        all_vars = tf.trainable_variables()
        for j, k in enumerate(zip(all_vars, gradvar)):
            if j % 2 == 0:
                var, gv = k
                g, v = gv
                v = get_top_k_array(v, 0.1)
                var.load(v, sess)
        if i % 100 == 0:
            print("The %s-th steps, loss = %f" % (i, loss))

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("test accuracy:", accuracy.eval({features: mnist.test.images, labels: mnist.test.labels}))