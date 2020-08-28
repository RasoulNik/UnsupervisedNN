# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:31:59 2020

@author: nikbakht
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from Loss_uplink import Loss
# from convNN import xNN
from xNN import xNN
class UNN(Layer):
    def __init__(self,Nap,Nuser,cost_type, **kwargs):
        super(UNN, self).__init__(**kwargs)
        self.Nap=Nap
        self.Nuser=Nuser
        self.cost_type = cost_type
    def build(self,input_shape):
        self.Network=xNN(self.Nuser)
        self.Loss=Loss(self.Nap,self.Nuser,self.cost_type)
    @tf.function
    def call(self,xin,SNR):
        p = self.Network(xin)
        cost,SIR,min_SINR = self.Loss(SNR,p)
        return cost,SIR,min_SINR
    # def debug(self,xin):
    #     with tf.GradientTape() as tape:
    #             # Forward pass.
    #             p = self.Network(xin)
    #             # Get gradients of loss wrt the weights.
    #             gradients = tape.gradient(p,self.variables)
    #     return gradients
