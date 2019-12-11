#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:19:17 2019

@author: rangeet
"""

import numpy as np
import datetime

class Util:
    def __init__(self, input_layer, hidden_layer, output_layer, lr):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.lr = lr
        # To-do
        self.input_W = 0.01*np.random.randn(input_layer, hidden_layer)
        self.input_B = 0.01*np.random.randn(hidden_layer)
        self.hidden_W = 0.01*np.random.randn(hidden_layer, output_layer)
        self.hidden_B = 0.01*np.random.randn(output_layer)
        self.regularizer = 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)  # only difference

    def forwardPropagation(self, inputs, label):
        """ Inputs are taken as the 1D array and so as the label"""
        node_hidden = np.dot(inputs, self.input_W)
        node_hidden = np.add(node_hidden, self.input_B)
        node_hidden = np.maximum(0, node_hidden)
        node_output = np.dot(node_hidden, self.hidden_W)
        node_output = np.add(node_output, self.hidden_B)
        #print(node_output)
        exp_node_output = np.exp(node_output)
        node_output = exp_node_output / np.sum(exp_node_output, axis=1, keepdims=True)
        #print(node_output)
        #node_output = self.softmax(node_output)
        loss = np.sum(-np.log(node_output[range(inputs.shape[0]),label]))/(inputs.shape[0])+0.5 * self.regularizer*np.sum(self.input_W *self.input_W)+0.5 * self.regularizer*np.sum(self.hidden_W *self.hidden_W)
        """Loss= Input data loss + Loss correction by penalizing the loss, here we use 0.2 as an experimental value"""
        #loss = np.sum(-np.log(node_output[range(inputs.shape[0]), label])) / (inputs.shape[0]) + 0.2 * self.regularizer * np.sum(self.input_W ^ 2) + 0.2 * self.regularizer * np.sum(self.hidden_W ^ 2)
        return loss, node_hidden, node_output

# =============================================================================
#     def forwardPropagation(self, inputs, label, input_W, input_B, hidden_W, hidden_B):
#         """ Inputs are taken as the 1D array and so as the label"""
#         node_hidden = np.dot(input, input_W)
#         node_hidden = np.add(node_hidden, input_B)
#         node_hidden = np.maximum(0, node_hidden)
#         node_output = np.dot(node_hidden, hidden_W)
#         node_output = np.add(node_output, hidden_B)
#         node_output = self.softmax(node_output)
#         """Loss= Input data loss + Loss correction by penalizing the loss, here we use 0.2 as an experimental value"""
#         loss = np.sum(-np.log(node_output[inputs.shape[0], label])) / (
#             inputs.shape[0]) + 0.2 * self.regularizer * np.sum(input_W ^ 2) + 0.2 * self.regularizer * np.sum(
#             hidden_W ^ 2) + 0.2 * self.regularizer * np.sum(input_B ^ 2) + 0.2 * self.regularizer * np.sum(
#             hidden_B ^ 2)
#         return loss, node_hidden, node_output
# =============================================================================

    def backwardPropagation(self, inputs, label, loss, node_hidden, node_output):
        """Initialize the error with the output value"""
        err = node_output
        err[range(inputs.shape[0]), label] -= 1
        err = err / inputs.shape[0]
        """Back propagate to hidden layer"""
        del_output_W = np.dot(node_hidden.T, err)
        """Back propagate to input layer"""
        del_input_W = np.dot(err, self.hidden_W.T)
        """ Use Relu function"""
        del_input_W[node_hidden <= 0] = 0
        del_input_W = np.dot(inputs.T, del_input_W)
        """Penalize the error with regularizer value"""
        del_input_W = del_input_W + self.regularizer * self.input_W
        del_output_W = del_output_W + self.regularizer * self.hidden_W
        """Store the error value into the weight value"""
        self.input_W += -self.lr * del_input_W
        self.hidden_W += -self.lr * del_output_W
    def accuracyComputation(self,inputs,label):
        inputs_to_layer_1=np.dot(inputs,self.input_W)
        inputs_to_layer_1=np.add(inputs_to_layer_1,self.input_B)
        layer_1=np.maximum(0,inputs_to_layer_1)
        layer_2=np.dot(layer_1, self.hidden_W)
        layer_2=np.add(layer_2,self.hidden_B)
        prediction=np.argmax(self.softmax(layer_2),axis=1)
        #print(prediction)
        return np.mean(prediction==label)


    def localSearch(self, input, label):
        min_loss = self.forwardPropagation(input, label, self.input_W, self.input_B, self.hidden_W, self.hidden_B)[0]
        #node_hidden = self.forwardPropagation(input, label, self.input_W, self.input_B, self.hidden_W, self.hidden_B)[1]
        #node_output = self.forwardPropagation(input, label, self.input_W, self.input_B, self.hidden_W, self.hidden_B)[2]
        delta = 0.01
        n = 0
        current_time = datetime.datetime.now()
        efficiency = False

        while True:
            loss = self.forwardPropagation(input, label, 0 + delta * n, 0 + delta * n, 0 + delta * n, 0 + delta * n)[0]
            if loss < min_loss:
                efficiency = True
                min_loss = loss
            after_search = datetime.datetime.now()
            if current_time - after_search == 3600:
                return delta * n, min_loss, efficiency
            n += 1
            # loss2 = self.forwardPropagation(input, label, 0 + delta * n, 0 + delta * n, 0 + delta * n, 0 + delta * n)[0]
# =============================================================================
# Not working code: Has code error. Will add after fix
#     def ASASearch(self, input_W, input_B):
#          delta=C
#          exit_flag=False
#          for i in self.inputs:
#              W_i= 0
#              B_i= 0
# #        label: check
#          while exit_flag ==False:
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
#  #             else:  
#  #                 goto check
#              W_p = W
#              B_p = B
#         #return 1-loss(W,B)
# =============================================================================
     
#    def loss(input_W, input_B):
         
         
#         return
         