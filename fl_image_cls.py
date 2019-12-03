# -*- coding: utf-8 -*-

import random
import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff

warnings.simplefilter('ignore')

np.random.seed(0)
# if six.PY3:
# 	tff.framework.set_default_executor(tff.framework.create_local_executor())

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
	emnist_train.client_ids[0])

# example_element = iter(example_dataset).next()

NUM_CLIENTS = 100
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
SELECTED_CLIENT_NUM = 50

def preprocess(dataset):

	def element_fn(element):
		return collections.OrderedDict([
			('x', tf.reshape(element['pixels'], [-1])),
			('y', tf.reshape(element['label'], [1])),
			])

	return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
		SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(
	lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

# print(sample_batch)

def make_federate_data(client_data, client_ids):
	return [preprocess(client_data.create_tf_dataset_for_client(x))
	for x in client_ids]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

# data in clients participate in training
federate_train_data = make_federate_data(emnist_train, sample_clients)

# data in clients participate in updating
selected_client = random.sample(range(NUM_CLIENTS), SELECTED_CLIENT_NUM)

selected_client_data = [federate_train_data[i] for i in selected_client]
# import pdb; pdb.set_trace()

def create_compiled_keras_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(
			10, activation='softmax', kernel_initializer='zeros', input_shape=(784,))])

	model.compile(
		loss = 'sparse_categorical_crossentropy',
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.02),
		metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
	return model

def model_fn():
	keras_model = create_compiled_keras_model()
	return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()
for i in range(2, 11):
	state, metrics = iterative_process.next(state, selected_client_data)
	print("Round {:2d}, metrics = {}".format(i, metrics))

