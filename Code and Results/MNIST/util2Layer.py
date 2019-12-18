#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:52:16 2019

@author: rangeetpan
"""

import tensorflow as tf


class Util:
    def __init__(self, n_input, n_classes, n_hidden_1, n_hidden_2):
        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Create model
    def multilayer_perceptron(self, x):
        # Hidden fully connected layer with 256 neurons
        # weights['h1'].assign_add(d)
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    # Construct model
    def train(self, x_train, y_train, learning_rate, training_epochs, batch_size):
        logits = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(600)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = x_train[100 * i:100 * (i + 1)]
                    batch_y = y_train[100 * i:100 * (i + 1)]
                    # batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            return avg_cost, accuracy.eval({self.X: x_train, self.Y: y_train})
