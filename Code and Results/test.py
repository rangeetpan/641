#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 02:46:19 2019

@author: rangeetpan
"""


from __future__ import print_function

# Import MNIST data
# =============================================================================
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# =============================================================================
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
from newutil import Util
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train=x_train.astype('float32')/255.0
x_train=x_train.reshape(60000,784)
y_train=to_categorical(y_train)
y_train=y_train.astype('float64')

# Parameters
count=0
learning_rate = 0.001
training_epochs = 10
batch_size = 100
delta1=[ [ +0.002 ] * 256 ] * 784
delta2=[ [ +0.002 ] * 256 ] * 256 # change delta
exit_flag=False
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
NN=Util(n_input,n_classes,n_hidden_1,n_hidden_2)
loss, accuracy=NN.train(x_train,y_train,learning_rate,training_epochs,batch_size)
print("Actual Accuracy: "+ str(accuracy))
while exit_flag ==False:
    NN.weights['h1'].assign_add(delta1)
    NN.weights['h2'].assign_add(delta2)
    #NN.biases['b1'].assign_add(delta1)
    #NN.biases['b2'].assign_add(delta2)
    loss1, accuracy=NN.train(x_train,y_train,learning_rate,training_epochs,batch_size)
    print("Trial "+str(count)+" ADNN Accuracy: "+str(accuracy))
    count=count+1
    obj=loss-loss1
    loss=loss1
    if(obj<=0):
        exit_flag ==False
    else:
        P = np.exp((-obj/loss1))
        P_rand = np.random.uniform(0,1)
        if P_rand <= P:
            exit_flag=False
        else:
            exit_flag=True