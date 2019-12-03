# -*- coding: utf-8 -*-

from six.moves import range
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation.models import mnist

# Load simulation data
source, _ = tff.simulation.datasets.emnist.load_data()
def client_data(n):
	dataset = source.create_tf_dataset_for_client(source.client_ids[n])
	return mnist.keras_dataset_from_emnist(dataset).repeat(10).batch(20)

# Pick a subset of client devices to participate in training
train_data = [client_data(n) for n in range(10)]


# Grab a single batch of data so that TFF knows what data looks like
sample_batch = tf.nest.map_structure(
	lambda x: x.numpy(), iter(train_data[0]).next())

# Wrap a keras model for use with TFF
def model_fn():
	return tff.learning.from_compiled_keras_model(
		mnist.create_simple_keras_model(), sample_batch)

# Simulate a few rounds of training with the selected client devices
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()
for i in range(5):
	state, metrics = trainer.next(state, train_data)
	print("round: ", i, "loss: ", metrics.loss)
# import pdb; pdb.set_trace()