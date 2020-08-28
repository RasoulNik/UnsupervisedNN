# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:53:00 2020

@author: nikbakht
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer
# import tensorflow_probability as tfp

class xNN(Layer):
    def __init__(self,Nuser,**kwargs):
        super(xNN, self).__init__(**kwargs)
        self.Nuser=Nuser
        
    def build(self,input_shape):
#
                
        self.dense0 = tf.keras.layers.Dense(units=1200,activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=300,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.Nuser)
#     def call(self,xin):
#         y=self.dense0(xin)
#         y=self.dense1(y)
#         # y_res
#         y = tf.concat([y,xin],axis=1)
#         y=self.dense2(y)
#         return y 
    # @tf.function
    def call(self,xin):
        y = self.dense0(xin)
        y = self.dense1(y)
        y = self.dense2(y)
        y = tf.exp(y)
        # p=p+1e-5
        return y
#     def call(self,xin):
#         with tf.GradientTape() as tape:
#             tape.watch(xin)
#             y=self.dense0(xin)
#             y=self.dense1(y)
#             y=self.dense2(y)
#             # y = tf.nn.relu(y)
#             # y = y+1e-5
#         grad = tape.gradient(y,self.trainable_variables)
#         check_grad =[0]*len(grad)
#         for i in range(len(grad)):
#             grad_check = tf.debugging.check_numerics(grad[i],'xNN: Gradient error')
#         # if tf.math.is_nan(grad[0]).numpy().flatten()[0]:
#         #     print('grad is fucked up')
#         # y = tf.math.sigmoid(y)
#         # y = tfp.math.clip_by_value_preserve_gradient(y,-10,10)
#         # y = tf.exp(y)
#         return y 