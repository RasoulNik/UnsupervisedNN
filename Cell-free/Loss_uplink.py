# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:20:46 2020

@author: nikbakht
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
#import numpy as np
class Loss(Layer):
    def __init__(self,Nap,Nuser,cost_type,**kwargs):
        super(Loss, self).__init__(**kwargs)
        self.Nap=Nap
        self.Nuser=Nuser
        self.alpha=1
        self.cost_type = cost_type
    def build(self,input_shape):
        pass

    @tf.function
    def call(self,SNR,p):
        # p = tf.math.exp(p)
    #    p=p+1e-5;
        num=tf.zeros([p.shape[0],1], dtype='float32') 
        denom=tf.zeros(num.shape, dtype='float32') 
        SINR=tf.zeros([SNR.shape[0],self.Nuser], dtype='float32')

        num = p*tf.square(tf.reduce_sum(SNR,axis=1))
        cross_gain =tf.linalg.matmul(SNR,SNR,transpose_a=True)
        denom = tf.squeeze(tf.linalg.matmul(cross_gain,tf.expand_dims(p,axis=2)))
        denom = denom -tf.linalg.diag_part(cross_gain)*p                         
        SINR = num/denom

        if self.cost_type == 'maxmin':
            temp = self.alpha * tf.pow(tf.divide(1.0, 0.01 + SINR), 0.4)
            Cost = tf.reduce_sum(tf.exp(temp), axis=1, keepdims=True)  # +
            Const = tf.reduce_sum(tf.nn.relu(p-1),axis=1,keepdims=True)
            Cost = 1/self.Nuser*(Cost+0.1*Const)

        elif self.cost_type == 'maxproduct':
            temp = tf.pow(tf.divide(1.0, 0.01 + SINR), .4)
            Cost = tf.reduce_sum(tf.math.log(0.01 + temp),axis=1)
            Const = tf.reduce_sum(tf.nn.relu(p - 1), axis=1, keepdims=True)
            Cost = 1 / self.Nuser * (Cost + .1 * Const)
        Cost = tf.reduce_mean(Cost, axis=0)
        min_SINR = tf.reduce_mean(tf.math.log(tf.math.reduce_min(SINR, axis=1)))
        return Cost,SINR,min_SINR


        