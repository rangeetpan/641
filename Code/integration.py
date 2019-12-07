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
NN = Util(784, 100, 10, 0.2)
for i in range(100):
    loss, node_hidden, node_output=NN.forwardPropagation(x_train, y_train)
    #print(loss)
    NN.backwardPropagation(x_train,y_train,loss,node_hidden, node_output)
    accuracy=NN.accuracyComputation(x_train,y_train)
    print("Iteration: "+str(i)+" Accuracy: "+str(accuracy))
# =============================================================================
# for ii in range(200):
#     loss = network.train(mnist.train.images, mnist.train.labels)
#     train_accuracy=network.run(mnist.train.images,mnist.train.labels)
#     test_accuracy=network.run(mnist.test.images,mnist.test.labels)
#     validation_accuracy=network.run(mnist.validation.images,mnist.validation.labels)
#     losses['train'].append(loss)
#     losses['train_accuracy'].append(train_accuracy)
#     losses['test_accuracy'].append(test_accuracy)
#     losses['validation_accuracy'].append(validation_accuracy)
#     if ii % 10 ==0:
#         print('interation %d --loss %f -- train accuracy:%.2f ----test accuracy: %.2f ---- validation accuracy: %.2f' % (ii,loss,train_accuracy,test_accuracy,validation_accuracy))
# =============================================================================
