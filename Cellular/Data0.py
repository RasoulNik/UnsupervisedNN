# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:39:18 2020

@author: nikbakht
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Data(Layer):
    def __init__(self,Nuser, **kwargs):
        super(Data, self).__init__(**kwargs)
        self.EX=100
        self.EY=100
        self.exponent=3.8
        self.shadowing_sigma=0;
        self.Zuser=0;
        self.Zap=1;
        self.Nuser_drop=10*Nuser
        self.Nap=Nuser
        self.Nuser=Nuser
        
    def call(self,batch_num,beta_open_loop=1):
        self.batch_num= batch_num
        batch_num = batch_num*2
        # Xin = tf.zeros([batch_num,2*(self.Nuser+self.Nap)],dtype='float32')
        G = tf.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
        power_propotional = tf.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
        x0 = tf.random.uniform([batch_num,self.Nuser_drop,1],0,self.EX)
        y0 = tf.random.uniform([batch_num,self.Nuser_drop,1],0,self.EY)
        z0 = self.Zuser+tf.zeros([batch_num,self.Nuser_drop,1],dtype='float32')
        Xuser = tf.concat([x0,y0,z0],axis=2)

        x = tf.random.uniform([batch_num,self.Nap,1],0,self.EX)
        y = tf.random.uniform([batch_num,self.Nap,1],0,self.EY)
        z = self.Zap+tf.zeros([batch_num,self.Nap,1],dtype='float32')  
        Xap = tf.concat([x,y,z],axis=2)

        d = self.Dist(Xap,Xuser,self.EX,self.EY)
    
        D_assign =self.Assign_AP(d)

        g = -46-10*self.exponent*tf.math.log(D_assign)/tf.math.log(10.0)
        g= g+self.shadowing_sigma*tf.random.normal([D_assign.shape[0],self.Nap,self.Nuser],0,1)
        g_linear=tf.pow(10.0,g/10)
        G = g_linear
        power_propotional=1/tf.pow(tf.linalg.diag_part(g_linear),beta_open_loop)
        # else:
        #     print('Not enough valid batches created')
        return G, power_propotional
    def Assign_AP(self,D):
        D_assign=tf.zeros([D.shape[0],self.Nap,1],dtype='float32')
        d_sort=tf.math.argmin(D,axis=1)
        # d_sort =tf.squeeze(d_sort)
        # Status=1
        # Make sure mask does not have zero value!!!!!!
        mask = tf.expand_dims(tf.range(1.0,self.Nuser_drop+1),axis=0)
        mask = tf.tile(mask,[D.shape[0],1])
        for i in range(self.Nap):
            
            # ind_i=np.argwhere(d_sort==i)
            # idnof user assigned to AP+i

            #----how many users assigned to AP_i
            ind_ap_i =d_sort ==i
            # compute valid batch (AP_i has atleast one user assigned)
            valid_batch = ind_ap_i
            valid_batch = tf.reduce_sum(tf.cast(valid_batch,'float32'),axis=1)
            valid_batch = tf.squeeze(tf.where(valid_batch>0))
            #-----------Keep valid batch
            # ind_i_val = tf.gather(ind_i_val,valid_batch)
            ind_ap_i = tf.gather(ind_ap_i,valid_batch,axis=0)
            d_sort = tf.gather(d_sort,valid_batch,axis=0)
            D = tf.gather(D,valid_batch,axis=0)
            D_assign = tf.gather(D_assign,valid_batch,axis=0)
            mask = tf.gather(mask,valid_batch,axis=0)
            #---------------------------------------------------
            mask_i =mask*tf.cast(ind_ap_i,'float32')
            mask_i = tf.math.round(tf.nn.softmax(100*mask_i,axis=1))
            mask_i = tf.tile(tf.expand_dims(mask_i,axis=1),[1,self.Nap,1])
            dist_selected_user = tf.reduce_sum(mask_i*D,axis=2, keepdims=True)
            # if tf.reduce_sum(tf.cast(dist_selected_user==0.0,'float32')):
            #     print('User assign error')
                                
            D_assign = tf.concat([D_assign,dist_selected_user],axis= 2)

        D_assign = D_assign[0:self.batch_num,:,1:]
        return D_assign

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
