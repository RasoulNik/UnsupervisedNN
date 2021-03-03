# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:56:46 2020

@author: nikbakht
"""
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from Data import Data
from UNN_downlink import UNN
import pickle
class Plot():
    def __init__(self,**kwargs):
        super(Plot, self).__init__(**kwargs)
    def call(self,x):
        self.cdfplot(x)
        
    def cdfplot(self,x):
        fig, ax = plt.subplots(1, 1)
        for i in range(len(x)):
            qe, pe =self.ecdf(10*np.log10(x[i].flatten()))
            ax.plot(qe, pe, lw=2, label=str(i))
            
        # ax.hold(True)
        
        ax.set_xlabel('(SINR)dB')
        ax.set_ylabel('CDF')
        ax.legend(fancybox=True, loc='right')
    #    plt.xlim([-10,30])
        plt.ylim([0,1])
        plt.show()
    def ecdf(self,sample):
        # convert sample to a numpy array, if it isn't already
        sample = np.atleast_1d(sample)
        # find the unique values and their corresponding counts
        quantiles, counts = np.unique(sample, return_counts=True)
    
        # take the cumulative sum of the counts and divide by the sample size to
        # get the cumulative probabilities between 0 and 1
        cumprob = np.cumsum(counts).astype(np.double) / sample.size
        return quantiles, cumprob
    def sinr_av(self,SNR,p,Nap,Nuser,mode='Clip'):
        # p=tf.cast(p,'float32')
        # SNR=tf.constant(SNR.astype('float32'),dtype='float32')
        # p=tf.math.exp(p)
        if mode=='Clip':
            p=tf.clip_by_value(p,0,1)
    #    p=p+1e-5;
        Hloop=100
        num=tf.zeros([p.shape[0],1], dtype='float32') 
        denom=tf.zeros(num.shape, dtype='float32') 
        sinr=tf.zeros([SNR.shape[0],Nuser], dtype='float32')
        ta = tf.TensorArray(tf.float32, size=Nuser)
        Hreal=tf.random.normal([SNR.shape[0],Nap,Nuser,Hloop])
        Himag=tf.random.normal([SNR.shape[0],Nap,Nuser,Hloop])
        H=1/np.sqrt(2.0)*(tf.complex(Hreal,Himag))
        Habs=tf.math.abs(H)
        Habs=tf.cast(tf.math.real(Habs),'float32')
        for k in range(Nuser):
            num=tf.expand_dims(p[:,k]*SNR[:,k,k],axis=1)
            num=tf.tile(num,[1,Hloop])*tf.math.square(Habs[:,k,k,:])
            Total=tf.expand_dims(p*SNR[:,:,k],axis=2)
            Total=tf.tile(Total,[1,1,Hloop])*tf.math.square(Habs[:,:,k,:])
            denom=1+tf.reduce_sum(Total,axis=1)-Total[:,k,:]
            ta = ta.write(k,tf.reduce_mean(tf.divide(num,denom),axis=1))
        sinr = tf.transpose(ta.stack(),perm=[1,0])
        return sinr
    def SINR(self,SNR,p,Nuser):
        # p=tf.cast(p,'float32')
        # SNR=tf.constant(SNR.astype('float64'),dtype='float64')
        # p=tf.math.exp(p)
    #    p=p+1e-5;
        num=tf.zeros([p.shape[0],1], dtype='float32') 
        denom=tf.zeros(num.shape, dtype='float32') 
        SINR=tf.zeros([SNR.shape[0],Nuser], dtype='float32')
        ta = tf.TensorArray(tf.float64, size=Nuser)
    
        for k in range(Nuser):
           num=tf.multiply(p[:,k],SNR[:,k,k])
           Total=tf.multiply(p,SNR[:,:,k])
           denom=1+tf.reduce_sum(Total,axis=1)-Total[:,k]
           ta = ta.write(k,tf.divide(num,denom))
    
        SINR=tf.transpose(ta.stack(),perm=[1,0])
        # temp=self.alpha*tf.pow(tf.divide(1.0,0.01+SINR),.4)
        # # Cost=1/self.Nuser*tf.math.reduce_logsumexp(temp,axis=1,keepdims=True)
        # Cost=1/self.Nuser*(tf.reduce_sum(tf.exp(temp),axis=1,keepdims=True))#+
        # # .1*tf.reduce_mean(tf.nn.relu(p-1),axis=1,keepdims=True)) 
        # Cost=tf.reduce_mean(Cost,axis=0)
        return SINR
