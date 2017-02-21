from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

label_keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def showImage(data, guess, answer):
	arr = np.array(data, dtype='float')
	arr = arr.reshape((28, 28))
	
	plt.title("Guess={guess}|Answer={answer}".format(guess=guess, answer=answer))
	plt.imshow(arr, cmap='gray')
	plt.show()

#Settings
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

#NN Parameters
n_hidden_1 = 256 # Layer 1 has 256 neurons
n_hidden_2 = 256 # Layer 2 has 256 neurons
n_input = 784 # Input is 28*28 = 784
n_classes = 10 # 10 possible outputs (0-9 digits)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c/total_batch

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	print("Optimization Finished!")
	
	print("Beginning evaluation...")
	guesses = tf.argmax(pred, 1).eval({x: mnist.test.images, y: mnist.test.labels})
	answers = tf.argmax(y, 1).eval({x: mnist.test.images, y: mnist.test.labels})

	for i in range(total_batch):
		batch_x = mnist.test.images;
		guess = label_keys[guesses[i]]
		answer = label_keys[answers[i]]
		showImage(batch_x[i], guess, answer) 

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
