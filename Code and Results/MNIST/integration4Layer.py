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
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from util4Layer import Util
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.astype('float32')/255.0
x_train=x_train.reshape(60000,784)
y_train=to_categorical(y_train)
y_train=y_train.astype('float64')
for i in range(0,3):
# Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256
    n_input = 784
    count=0
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100
    delta=0.0055+i*0.0005
    delta1=[ [ +delta ] * n_hidden_1 ] * n_input
    delta2=[ [ +delta ] * n_hidden_2 ] * n_hidden_1 # change delta
    delta3=[ +delta ] * n_hidden_1
    delta4= [ +delta ] * n_hidden_2   # change delta
    exit_flag=False
    n_hidden_3 = 256 # 3rd layer number of neurons
    n_hidden_4 = 256 # 4th layer number of neurons
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    NN=Util(n_input,n_classes,n_hidden_1,n_hidden_2,n_hidden_3,n_hidden_4)
    loss, accuracy=NN.train(x_train,y_train,learning_rate,training_epochs,batch_size)
    text_file = open("Output_mnist_Layer4_"+str(delta)+".txt", "w")
    text_file.write("Actual Accuracy: "+ str(accuracy)+"\n")
    while count<=50:
        NN.weights['h1'].assign_add(delta1)
        NN.weights['h2'].assign_add(delta2)
        NN.weights['h3'].assign_add(delta2)
        NN.weights['h4'].assign_add(delta2)
        NN.biases['b1'].assign_add(delta3)
        NN.biases['b2'].assign_add(delta4)
        NN.biases['b3'].assign_add(delta3)
        NN.biases['b4'].assign_add(delta4)
        loss1, accuracy=NN.train(x_train,y_train,learning_rate,training_epochs,batch_size)
        text_file.write("Trial "+str(count)+" ADNN Accuracy: "+str(accuracy)+"\n")
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
    text_file.close()