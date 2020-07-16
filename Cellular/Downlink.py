# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:36:42 2020

@author: nikbakht
"""
from __future__ import absolute_import, division, print_function
#---------------------------------
import tensorflow as tf
import socket
host=socket.gethostname()
if host=='N-20HEPF10AVE2':
    pass
else:
    num_CPU = 1:4 # GPU  to use, can be 0, 2
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    cpus = tf.config.experimental.list_physical_devices("CPU")
    print('Number of GPUs available :', len(gpus))
    tf.config.experimental.set_visible_devices(cpus[num_GPU], 'CPU')
    tf.config.experimental.set_memory_growth(cpus[num_CPU], mem_growth)
    print('Used CPU: {}. Memory growth: {}'.format(num_CPU, mem_growth))
    #-----------------------------------------------------------------
import numpy as np
# import matplotlib.pyplot as plt
#import scipy.io as sio
#import h5py
#import pandas as pd
from datetime import datetime
# from Data_conv import Data
from Data import Data
from Plot_results_downlink import Plot

# from UNNdebug import UNN
from UNN import UNN
from Loss_downlink import Loss
import pickle

#------------------------------------------
tf.keras.backend.set_floatx('float64')
#train_iterations = 100
batch_size =200
# train_per_database=100
# database_size=batch_size*train_per_database
EPOCHS =int(5e3)
Nuser=30
Nap=30
#Lambda=.001
#alpha=1
Id_save='2'
save_model=1
P_over_noise=120; # dB
cost_type='maxmin'
# cost_type='maxproduct'
#-----------------------------------------

def train(obj,Dataobj,epochs,mode):
    # TF board logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    best_test_rate = -float('inf')
    best_W = None
    LR=np.logspace(-3,-4, num=epochs)
    Xin,G_batch,_=Dataobj(100*batch_size)
    SNR=np.power(10,P_over_noise/10)*G_batch
    Xin=np.power(10,P_over_noise/10)*tf.linalg.diag_part(SNR)
#     Xin=np.power(10,P_over_noise/10)*tf.reshape(SNR,[SNR.shape[0],-1])
    obj.Xin_av=np.mean(np.log(Xin),axis=0)
    obj.Xin_std=np.std(np.log(Xin),axis=0)
    try:
        for i in range(epochs):
            LR_i=LR[i ]
            optimizer = tf.keras.optimizers.Adam(LR_i)
            xin,G_batch,_=Dataobj(batch_size)
            SNR=np.power(10,P_over_noise/10)*G_batch
            # xin=np.reshape(G_batch,[batch_size,-1])
            xin=np.log(np.diagonal(SNR,axis1=1,axis2=2))
#             xin=np.power(10,P_over_noise/10)*tf.reshape(SNR,[SNR.shape[0],-1])
            xin=(xin-obj.Xin_av)/obj.Xin_std
            J=[]
            with tf.GradientTape() as tape:
                # Forward pass.
                cost,_ = obj(xin,SNR)
                # Get gradients of loss wrt the weights.
                gradients = tape.gradient(cost, obj.trainable_weights)
                # Gradient clipping
#                 c_gradients,grad_norm = tf.clip_by_global_norm(gradients, 1.0)
                # Update the weights of our linear layer.
                optimizer.apply_gradients(zip(gradients, obj.trainable_weights))
                J.append(cost.numpy())
            
    
            if i % 50 == 0:
                # test_rate=cost.numpy()[0]
                test_rate=np.mean(J)
#                bit2r.LR=bit2r.LR*.85
                print('iter i=',i,'average cost is ', test_rate)
                if test_rate > best_test_rate:
                    best_test_rate = test_rate
                best_W = obj.get_weights()
                save_model(obj, 'models/'+'Downlink'+mode+'UNN.mod')

                with train_summary_writer.as_default():
                    tf.summary.scalar('test rate', test_rate, step=i)
                    tf.summary.scalar('best test rate', best_test_rate, step=i)
                
    except KeyboardInterrupt:
        pass
    
    obj.set_weights(best_W)
    return

def save_model(model, fn):
    W = model.get_weights()
    with open(fn, 'wb') as f:
        pickle.dump(W, f)
        
def load_model(model, fn):
    with open(fn, 'rb') as f:
        W = pickle.load(f)
    model.set_weights(W)
def SINR(SNR,p,Nap,Nuser):
    p=tf.exp(p)
#    p=p+1e-5;
    num=tf.zeros([p.shape[0],1], dtype='float64') 
    denom=tf.zeros(num.shape, dtype='float64') 
    SINR=tf.zeros([SNR.shape[0],Nuser], dtype='float64')
    ta = tf.TensorArray(tf.float64, size=Nuser)

    for k in range(Nuser):
       num=tf.multiply(p[:,k],SNR[:,k,k])
       Total=tf.multiply(p,SNR[:,k,:])
       denom=1+tf.reduce_sum(Total,axis=1)-Total[:,k]
       ta = ta.write(k,tf.divide(num,denom))

    SINR=tf.transpose(ta.stack(),perm=[1,0])
    return SINR
######    
data=Data(Nuser)
unn=UNN(Nap,Nuser,cost_type)
train(unn,data,EPOCHS,cost_type)
#tensorboard --logdir ./logs --bind_all
#---------------------------------------------
# load=0
xin,G_batch,p_frac=data(2*batch_size,.5)
# xin=np.reshape(G_batch,[batch_size,-1])
SNR=np.power(10,P_over_noise/10)*G_batch
xin=np.log(np.diagonal(SNR,axis1=1,axis2=2))
xin=(xin-unn.Xin_av)/unn.Xin_std
# if load:
#    unn=UNN(Nap,Nuser,cost_type) 
#    cost,_ = unn(xin,SNR)
#    load_model(unn, 'C:\\Users\\nikbakht\\OneDrive - Nokia\\UPF\\Codes\\UNN\\Cellular\\python\\lib\\models\\xUNN.mod')
  
# xin=(xin-unn.Xin_av)/unn.Xin_std
RP=Plot()
SIR_NN=RP.sinr_av(SNR,unn.Network(xin),Nap,Nuser)
SIR_frac=RP.sinr_av(SNR,tf.math.log(p_frac),Nap,Nuser)
sir_vec=[SIR_NN.numpy(),SIR_frac.numpy()]
RP.cdfplot(sir_vec)