from __future__ import print_function

from sklearn.model_selection import train_test_split

import os 
import numpy as np
import tensorflow as tf

def unpack_data():

	return np.load("Datasets/DS1.dat")

def split_data(data):

	X = []
	y = []

	for i in data:
		X.append(i[0])
		y.append(i[1])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return(X_train, X_test, y_train, y_test)

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_1, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return out_layer

def Train(X_train, X_test, y_train, y_test):

	learning_rate = 0.001
	training_epochs = 15
	batch_size = 100
	display_step = 1

	# Network Parameters
	n_hidden_1 = 2048 # 1st layer number of features
	n_hidden_2 = 2048 # 2nd layer number of features
	n_hidden_3 = 2048 # 3rd layer number of features
	n_input = 768 # 8x8x12 64 is the size of the board, 12 different pieces W and B
	n_classes = 1 # 1 output = evaluation f(x) value

	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])
	
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

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.global_variables_initializer()

if __name__ == '__main__':
	
	Ds = unpack_data()
	
	X_train = split_data(Ds)[0]
	X_test = split_data(Ds)[1]
	y_train = split_data(Ds)[2]
	y_test = split_data(Ds)[3]

	Train(X_train, X_test, y_train, y_test)
