#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:24:46 2019

@author: rangeetpan
"""
from util import Util
#from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
X_train=X_train.astype('float32')/255.0
X_train=X_train.reshape(60000,784)
NN = Util(784, 100, 10, 0.2)
for i in range(100):
    loss, node_hidden, node_output=NN.forwardPropagation(X_train, Y_train)
    #print(loss)
    NN.backwardPropagation(X_train,Y_train,loss,node_hidden, node_output)
    accuracy=NN.accuracyComputation(X_train,Y_train)
    print("Iteration: "+str(i)+" Accuracy: "+str(accuracy))

