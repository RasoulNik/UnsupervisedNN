# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:39:18 2020

@author: nikbakht
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Data(Layer):
    def __init__(self,Nap,Nuser, **kwargs):
        super(Data, self).__init__(**kwargs)
        self.EX=100
        self.EY=100
        self.exponent=3.8
        self.shadowing_sigma=0;
        self.Zuser=0;
        self.Zap=1;
        self.Nap=Nap
        self.Nuser=Nuser
    def call(self,batch_num,beta_open_loop=1):

        # Xin = tf.zeros([batch_num,2*(self.Nuser+self.Nap)],dtype='float32')
        G = tf.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
        power_propotional = tf.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
        # i=0;
        # while i<batch_num:
        x0 = tf.random.uniform([batch_num,self.Nuser,1],0,self.EX)
        y0 = tf.random.uniform([batch_num,self.Nuser,1],0,self.EY)
        z0 = self.Zuser+tf.zeros([batch_num,self.Nuser,1],dtype='float32')
        Xuser = tf.concat([x0,y0,z0],axis=2)
    #        Xin.append([reshape([x,y],[1,Nuser*2])])
        
        
        x = tf.random.uniform([batch_num,self.Nap,1],0,self.EX)
        y = tf.random.uniform([batch_num,self.Nap,1],0,self.EY)
        z = self.Zap+tf.zeros([batch_num,self.Nap,1],dtype='float32')  
        Xap = tf.concat([x,y,z],axis=2)
        
        # Xin[[i],:]=np.concatenate((np.transpose(x0),np.transpose(y0),np.transpose(x),np.transpose(y)),axis=1)
        
    #        d=Dist_no_wrap(Xap,Xuser,EX,EY)
        d = self.Dist(Xap,Xuser,self.EX,self.EY)
    
        #Compute Gains
        g = -46.0-10.0*self.exponent*tf.math.log(d)/tf.math.log(10.0)
        g = g+self.shadowing_sigma*tf.random.normal([batch_num,self.Nap,self.Nuser])
        g_linear = tf.pow(10.0,g/10.0)
        G = g_linear
        power_propotional = 1/tf.pow(tf.reduce_sum(G,axis=1),beta_open_loop)
    
        return G,power_propotional
    
    def Dist(self,X1,X2,EX,EY): 
        N1 = X1.shape[1]
        N2 = X2.shape[1]
        #----------The pair distances
        xvec1 = tf.expand_dims(X1[:,:,0],axis=2)
        xvec2 = tf.expand_dims(X2[:,:,0],axis=2)
        xmat1 = tf.tile(xvec1,[1,1,N2])
        xmat2 = tf.tile(tf.transpose(xvec2,perm=[0,2,1]),[1,N1,1])
        xdiff = xmat1-xmat2
        xdist2 = tf.pow(tf.math.minimum(tf.math.abs(xdiff),EX-tf.math.abs(xdiff)),2)
        
        yvec1 = tf.expand_dims(X1[:,:,1],axis=2)
        yvec2 = tf.expand_dims(X2[:,:,1],axis=2)
        ymat1 = tf.tile(yvec1,[1,1,N2])
        ymat2 = tf.tile(tf.transpose(yvec2,perm=[0,2,1]),[1,N1,1])
        ydiff = ymat1-ymat2
        ydist2 = tf.pow(tf.minimum(tf.math.abs(ydiff),EY-tf.math.abs(ydiff)),2)
        
        zvec1 = tf.expand_dims(X1[:,:,2],axis=2)
        zvec2 = tf.expand_dims(X2[:,:,2],axis=2)
        zmat1 = tf.tile(zvec1,[1,1,N2]);
        zmat2 = tf.tile(tf.transpose(zvec2,perm=[0,2,1]),[1,N1,1]);
        zdiff = zmat1-zmat2
        zdist2=tf.pow(zdiff,2)
        D=tf.math.sqrt(xdist2+ydist2+zdist2)
        return D
    # def call(self,batch_num,beta_open_loop=.5):
    #     Xin=np.zeros([batch_num,2*(self.Nuser+self.Nap)],dtype='float32')
    #     G=np.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
    #     power_propotional=np.zeros([batch_num,self.Nuser],dtype='float32')
    #     i=0;
    #     while i<batch_num:
    #         x0=np.random.uniform(0,self.EX, size=[self.Nuser_drop,1])
    #         y0=np.random.uniform(0,self.EY, size=[self.Nuser_drop,1])
    #         z0=self.Zuser+np.zeros([self.Nuser_drop,1],dtype='float32')
    #         Xuser=np.concatenate((x0,y0,z0),axis=1)
    # #        Xin.append([reshape([x,y],[1,Nuser*2])])
            
            
    #         x=np.random.uniform(0,self.EX, size=[self.Nap,1])
    #         y=np.random.uniform(0,self.EY, size=[self.Nap,1])
    #         z=self.Zap+np.zeros([self.Nap,1],dtype='float32')  
    #         Xap=np.concatenate((x,y,z),axis=1)
            
    # #        Xin[[i],:]=np.concatenate((np.transpose(x0),np.transpose(y0),np.transpose(x),np.transpose(y)),axis=1)
            
    # #        d=Dist_no_wrap(Xap,Xuser,EX,EY)
    #         d=self.Dist(Xap,Xuser)

    #         g=-46-10*self.exponent*np.log10(d)+self.shadowing_sigma*np.random.randn(self.Nap,self.Nuser)
    #         g_linear=np.power(10,g/10)
    #         G[i,:,:]=g_linear;
    #         power_propotional[i,:]=1/np.power(np.diagonal(g_linear),beta_open_loop)
    #         i=i+1 
    #     G=G.astype('float64')
    #     Xin=Xin.astype('float64')
    #     power_propotional=power_propotional.astype('float64')
    #     return Xin,G, power_propotional 
    # def Dist(self,X1,X2): 
    #     N1 = X1.shape[0]
    #     N2 = X2.shape[0]
    #     #----------The pair distances
    #     xvec1=X1[:,[0]]
    #     xvec2=X2[:,[0]]
    #     xmat1=np.tile(xvec1,[1,N2])
    #     xmat2=np.tile(np.transpose(xvec2),[N1,1])
    #     xdiff=xmat1-xmat2
    #     xdist2=np.power(np.minimum(np.abs(xdiff),self.EX-np.abs(xdiff)),2)
        
    #     yvec1=X1[:,[1]]
    #     yvec2=X2[:,[1]]
    #     ymat1=np.tile(yvec1,[1,N2])
    #     ymat2=np.tile(np.transpose(yvec2),[N1,1]);
    #     ydiff=ymat1-ymat2
    #     ydist2=np.power(np.minimum(np.abs(ydiff),self.EY-np.abs(ydiff)),2)
        
    #     zvec1=X1[:,[2]]
    #     zvec2=X2[:,[2]]
    #     zmat1=np.tile(zvec1,[1,N2]);
    #     zmat2=np.tile(np.transpose(zvec2),[N1,1]);
    #     zdiff=zmat1-zmat2;
    #     zdist2=np.power(zdiff,2)
    #     D=np.sqrt(xdist2+ydist2+zdist2)
    #     return D
