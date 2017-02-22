from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

label_keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def showImage(data, guess, answer, correct):
	arr = np.array(data, dtype='float')
	arr = arr.reshape((28, 28))
	
	plt.title("Guess={guess}|Answer={answer}|Correct={correct}".format(guess=guess, answer=answer, correct=correct))
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
n_hidden_3 = 256 # Layer 3 has 256 neurons
n_input = 784 # Input is 28*28 = 784
n_classes = 10 # 10 possible outputs (0-9 digits)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Function to create a 2 layer perceptron using relu activation
def multilayer_perceptron(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.relu(layer_3)

	out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
	return out_layer

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create the perceptron
pred = multilayer_perceptron(x, weights, biases)

# Cost function is avg softmax
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize saver object for restorting the model
saver = tf.train.Saver()

with tf.Session() as sess:
	print("Restoring model...")
	saver.restore(sess, './saved_models/mnist/model3.ckpt')
	print("Model restored.")

	print("Beginning evaluation of model...")
	guesses = tf.argmax(pred, 1).eval({x: mnist.test.images, y: mnist.test.labels})
	answers = tf.argmax(y, 1).eval({x: mnist.test.images, y: mnist.test.labels})
	correctness = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)).eval({x: mnist.test.images, y: mnist.test.labels})

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

	# Show all images w/ guess, answer, and correctness
	for i in range(len(mnist.test.images)):
		batch_x = mnist.test.images;
		guess = label_keys[guesses[i]]
		answer = label_keys[answers[i]]
		correct = correctness[i]
		showImage(batch_x[i], guess, answer, correct) 


