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
        Xin=np.zeros([batch_num,2*(self.Nuser+self.Nap)],dtype='float32')
        G=np.zeros([batch_num,self.Nap,self.Nuser],dtype='float32')
        power_propotional=np.zeros([batch_num,self.Nuser],dtype='float32')
        i=0;
        while i<batch_num:
            x0=np.random.uniform(0,self.EX, size=[self.Nuser_drop,1])
            y0=np.random.uniform(0,self.EY, size=[self.Nuser_drop,1])
            z0=self.Zuser+np.zeros([self.Nuser_drop,1],dtype='float32')
            Xuser=np.concatenate((x0,y0,z0),axis=1)
    #        Xin.append([reshape([x,y],[1,Nuser*2])])
            
            
            x=np.random.uniform(0,self.EX, size=[self.Nap,1])
            y=np.random.uniform(0,self.EY, size=[self.Nap,1])
            z=self.Zap+np.zeros([self.Nap,1],dtype='float32')  
            Xap=np.concatenate((x,y,z),axis=1)
            
    #        Xin[[i],:]=np.concatenate((np.transpose(x0),np.transpose(y0),np.transpose(x),np.transpose(y)),axis=1)
            
    #        d=Dist_no_wrap(Xap,Xuser,EX,EY)
            d=self.Dist(Xap,Xuser)
            D_assign,Status=self.Assign_AP(d)
            if Status:
                #Compute Gains
                g=-46-10*self.exponent*np.log10(D_assign)+self.shadowing_sigma*np.random.randn(self.Nap,self.Nuser)
                g_linear=np.power(10,g/10)
                G[i,:,:]=g_linear;
                power_propotional[i,:]=1/np.power(np.diagonal(g_linear),beta_open_loop)
                i=i+1 
        # G=G.astype('float64')
        # Xin=Xin.astype('float64')
        # power_propotional=power_propotional.astype('float64')
        return G, power_propotional
    def Dist(self,X1,X2): 
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        #----------The pair distances
        xvec1=X1[:,[0]]
        xvec2=X2[:,[0]]
        xmat1=np.tile(xvec1,[1,N2])
        xmat2=np.tile(np.transpose(xvec2),[N1,1])
        xdiff=xmat1-xmat2
        xdist2=np.power(np.minimum(np.abs(xdiff),self.EX-np.abs(xdiff)),2)
        
        yvec1=X1[:,[1]]
        yvec2=X2[:,[1]]
        ymat1=np.tile(yvec1,[1,N2])
        ymat2=np.tile(np.transpose(yvec2),[N1,1]);
        ydiff=ymat1-ymat2
        ydist2=np.power(np.minimum(np.abs(ydiff),self.EY-np.abs(ydiff)),2)
        
        zvec1=X1[:,[2]]
        zvec2=X2[:,[2]]
        zmat1=np.tile(zvec1,[1,N2]);
        zmat2=np.tile(np.transpose(zvec2),[N1,1]);
        zdiff=zmat1-zmat2;
        zdist2=np.power(zdiff,2)
        D=np.sqrt(xdist2+ydist2+zdist2)
        return D
    def Assign_AP(self,D):
        D_assign=np.zeros([self.Nap,self.Nap],dtype='float32')
        d_sort=np.argmin(D,axis=0)
        Status=1
        for i in range(self.Nap):
            ind_i=np.argwhere(d_sort==i)
    #        val_i=d_sort[ind_i]
            if not ind_i.size==0:
                user_assigned_AP_i=ind_i[np.random.randint(0,ind_i.size)]
                D_assign[:,i]=D[:,user_assigned_AP_i[0]]
            else:
                Status=0 
                return D_assign, Status
        return D_assign,Status 