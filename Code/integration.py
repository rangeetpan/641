#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:24:46 2019

@author: rangeetpan
"""
from util import Util
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.astype('float32')/255.0
x_train=x_train.reshape(60000,784)
NN = Util(784, 100, 10, 0.1)
count=1
##############Algorithm Based Initialization#################
delta=0.002
exit_flag=False
input_W = 0.008*np.random.randn(784, 100)
hidden_W = 0.008*np.random.randn(100, 10)
input_B = 0.008*np.random.randn(100)
hidden_B = 0.008*np.random.randn(10)
NN.input_W=input_W
NN.input_B=input_B
NN.hidden_W=hidden_W
NN.hidden_B=hidden_B
for i in range(100):
    loss, node_hidden, node_output=NN.forwardPropagation(x_train, y_train)
    #print(loss)
    NN.backwardPropagation(x_train,y_train,loss,node_hidden, node_output)
    accuracy=NN.accuracyComputation(x_train,y_train)
print("Actual Accuracy: "+str(accuracy))
while exit_flag ==False:
    input_W=input_W-delta
    input_B=input_B-delta
    hidden_W=hidden_W-delta
    hidden_B=hidden_B-delta
    NN.input_W=input_W
    NN.input_B=input_B
    NN.hidden_W=hidden_W
    NN.hidden_B=hidden_B
    for i in range(100):
        loss1, node_hidden, node_output=NN.forwardPropagation(x_train, y_train)
        #print(loss)
        NN.backwardPropagation(x_train,y_train,loss,node_hidden, node_output)
        accuracy=NN.accuracyComputation(x_train,y_train)
    print("Trial "+str(count)+" ADNN Accuracy: "+str(accuracy))
    count=count+1
    obj=loss-loss1
    loss=loss1
    if(obj<=0):
        exit_flag ==False
    else:
        P = np.exp((-obj/loss1))
        P_rand = np.random.normal()
        if P_rand <= P:
            exit_flag=False
#              W = W_i+delta
#              B = B_i+delta
#              if W in input_W and B in input_B:
#                  obj=loss(W_i,B_i)-loss(W,B)
#                  if obj <= 0:
#                      exit_flag=False
#                  else:
#                      P = np.exp((-obj/loss(W,B)))
#                      P_rand = np.random
#                      if P_rand <= P:
#                          exit_flag=False

