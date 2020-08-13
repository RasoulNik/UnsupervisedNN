# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer

class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        self.Nuser=Nuser
        
    def build(self,input_shape):
#
                
        self.dense0 = tf.keras.layers.Dense(units=500,activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=200,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.Nuser)
    
    def call(self,xin):
        y=self.dense0(xin)
        y=self.dense1(y)
        y=self.dense2(y)
        y = tf.math.exp(y)
        return y  