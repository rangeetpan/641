#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:19:17 2019

@author: rangeet
"""

import numpy as np
class Util:
    def __init__(self,input_layer, hidden_layer, output_layer, lr):
        self.input_layer=input_layer
        self.output_layer=output_layer
        self.lr=lr
        self.input_W=np.random.randn(input_layer,hidden_layer)
        self.input_B=np.random.randn(hidden_layer)
        self.hidden_W=np.random.randn(hidden_layer,output_layer)
        self.hidden_B=np.random.randn(output_layer)
        self.regularizer=0
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    def forwardPropagation(self, inputs, label):
        """ Inputs are taken as the 1D array and so as the label"""
        node_hidden=np.dot(input,self.input_W)
        node_hidden=np.add(node_hidden,self.input_B)
        node_hidden=np.maximum(0,node_hidden)
        node_output=np.dot(node_hidden, self.hidden_W)
        node_output=np.add(node_output,self.hidden_B)
        node_output=self.softmax(node_output)
        """Loss= Input data loss + Loss correction by penalizing the loss, here we use 0.2 as an experimental value"""
        loss=np.sum(-np.log(node_output[inputs.shape[0],label]))/(inputs.shape[0])+ 0.2*self.regularizer*np.sum(self.input_W^2) + 0.2*self.regularizer*np.sum(self.hidden_W^2)+0.2*self.regularizer*np.sum(self.input_B^2) + 0.2*self.regularizer*np.sum(self.hidden_B^2)
        return loss,node_hidden,node_output
    def backwardPropagation(self, inputs, label, loss, node_hidden, node_output):
        """Initialize the error with the output value"""
        err =node_output
        err[inputs.shape[0],label] -= 1
        err = err/inputs.shape[0]
        """Back propagate to hidden layer"""
        del_output_W=np.dot(node_hidden.T,err)
        """Back propagate to input layer"""
        del_input_W=np.dot(err, self.hidden_W.T)
        """ Use Relu function"""
        del_input_W[node_hidden<=0]=0
        del_input_W=np.dot(inputs.T, del_input_W)
        """Penalize the error with regularizer value"""
        del_input_W=del_input_W+self.regularizer*self.input_W
        del_output_W=del_output_W+self.regularizer*self.output_W
        """Store the error value into the weight value"""
        self.input_W+=-self.lr*del_input_W
        self.output_W+=-self.lr*del_output_W


