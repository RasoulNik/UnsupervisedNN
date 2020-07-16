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
    def call(self,SNR,p):
        # p=-tf.nn.relu(p)

        # p=p+1e-5
        p = tf.math.exp(p)
        num=tf.zeros([p.shape[0],1], dtype='float32') 
        denom=tf.zeros(num.shape, dtype='float32') 
        SINR=tf.zeros([SNR.shape[0],self.Nuser], dtype='float32')
        # ta = tf.TensorArray(tf.float32, size=self.Nuser)
    
        # for k in range(self.Nuser):
        #    num=tf.multiply(p[:,k],SNR[:,k,k])
        #    Total=tf.multiply(p,SNR[:,:,k])
        #    denom=1+tf.reduce_sum(Total,axis=1)-Total[:,k]
        #    ta = ta.write(k,tf.divide(num,denom))
    
        # SINR=tf.transpose(ta.stack(),perm=[1,0])
        num = p*tf.linalg.diag_part(SNR)
        Total = tf.tile(tf.expand_dims(p,axis=2),[1,1,self.Nuser])*SNR
        denom = tf.reduce_sum(Total,axis=1)-tf.linalg.diag_part(Total)
        SINR = num/denom
        # Cost=1/self.Nuser*tf.math.reduce_logsumexp(temp,axis=1,keepdims=True)
        if self.cost_type=='maxmin':
                temp=self.alpha*tf.pow(tf.divide(1.0,0.01+SINR),0.5)
                Cost=1/self.Nuser*(tf.reduce_sum(tf.exp(temp),axis=1,keepdims=True))#+
                # .1*tf.reduce_mean(tf.nn.relu(p-1),axis=1,keepdims=True)) 
                Cost=tf.reduce_mean(Cost,axis=0)
        elif self.cost_type=='maxproduct':
                temp=tf.pow(tf.divide(1.0,0.01+SINR),.4)
                Cost=tf.reduce_mean(tf.math.log(0.01 +temp)
                  # +.1*tf.reduce_mean(tf.nn.relu(p-1),axis=1,keepdims=True) 
                ,axis=1)
        # Cost=1/self.Nuser*(tf.reduce_sum(tf.exp(temp),axis=1,keepdims=True)+
        #  .1*tf.reduce_mean(tf.nn.relu(p-1),axis=1,keepdims=True)) 
        Cost=tf.reduce_mean(Cost,axis=0)
        if  tf.math.is_inf(Cost).numpy() or tf.math.is_nan(Cost).numpy():
            print('Cost is inf or nan')
        min_sir = tf.reduce_mean(tf.math.log(tf.math.reduce_min(SINR,axis=1)))
        return Cost,SINR,min_sir
