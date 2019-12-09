#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:24:46 2019

@author: rangeetpan
"""
from util import Util
from tensorflow.examples.tutorials.mnist import input_data
input_data = input_data.read_data_sets('MNIST_data')
x_train=input_data.train.images
y_train=input_data.train.labels
NN = Util(784, 100, 10, 0.1)
for i in range(400):
    loss, node_hidden, node_output=NN.forwardPropagation(x_train, y_train)
    #print(loss)
    NN.backwardPropagation(x_train,y_train,loss,node_hidden, node_output)
    accuracy=NN.accuracyComputation(x_train,y_train)
    print("Iteration: "+str(i)+" Accuracy: "+str(accuracy))

