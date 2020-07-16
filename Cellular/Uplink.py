# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:36:42 2020

@author: nikbakht
"""
#---------------------------------
import tensorflow as tf
#import socket
GPU_mode=0 # set this value one if you have proper GPU setup in your computer
#The easiet way for using the GPU is docker

if GPU_mode: 
    num_GPU = 0 # choose among available GPUs
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print('Number of GPUs available :', len(gpus))
    tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num_GPU], mem_growth)
    print('Used GPU: {}. Memory growth: {}'.format(num_GPU, mem_growth))
#---------------------------------------------------------------------
import os    
import numpy as np
import time 
# import matplotlib.pyplot as plt
import scipy.io as sio
#import h5py
#import pandas as pd
from datetime import datetime
# from Data_conv import Data
from Data0 import Data
from Plot_results import Plot

# from UNNdebug import UNN
from UNN_Uplink import UNN
# from Loss import Loss
import pickle

#------------------------------------------
# tf.keras.backend.set_floatx('float64')
#train_iterations = 100
batch_size = 100
# train_per_database=100
# database_size=batch_size*train_per_database
EPOCHS =int(10e3)
Nuser = 5
Nap = 5
#Lambda=.001
#alpha=1
Id_save='2'
P_over_noise=120 # dB
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
    G_batch,_=Dataobj(5*batch_size)
    SNR=np.power(10,P_over_noise/10)*G_batch
#     Xin=np.reshape(np.log(SNR),[SNR.shape[0],-1])
    Xin=tf.linalg.diag_part(tf.math.log(SNR))
    obj.Xin_av=np.mean(Xin,axis=0)
    obj.Xin_std=np.std(Xin,axis=0)
    J_total =[]
    min_SINR_total=[]
    try:
        for i in range(epochs):
            LR_i=LR[i ]
            optimizer = tf.keras.optimizers.Adam(LR_i)
            # 100*batch_size is the size of each small database
            G_batch,_=Dataobj(10*batch_size)
            SNR=tf.pow(10.0,P_over_noise/10.0)*G_batch
#             xin=tf.reshape(tf.math.log(SNR),[SNR.shape[0],-1])
            xin=tf.linalg.diag_part(tf.math.log(SNR))
            xin=(xin-obj.Xin_av)/obj.Xin_std
            J=[]
            min_SINR_vec =[]
            for j in range(20):
                index = tf.random.uniform([batch_size],0,xin.shape[0],dtype=tf.dtypes.int32)
                xin_j = tf.gather(xin,index,axis=0)
                SNR_j = tf.gather(SNR,index,axis=0)
                with tf.GradientTape() as tape:
                    # Forward pass.
                    cost,_,min_SINR = obj(xin_j,SNR_j)
                    # Get gradients of loss wrt the weights.
                    gradients = tape.gradient(cost, obj.trainable_weights)
                    # Gradient clipping
                    gradients,grad_norm = tf.clip_by_global_norm(gradients, 1.0)
                    
                    # Update the weights of our linear layer.
                    # grad_check = [0]*len(c_gradients)
                    # for grad_i in range(len(c_gradients)):
                    #     # try:
                    #     grad_check = tf.debugging.check_numerics(c_gradients[grad_i],'UNN: Gradient error')
                    #     # except:
                    #     #     pass
                    # with tf.control_dependencies([grad_check]):
                optimizer.apply_gradients(zip(gradients, obj.trainable_weights))
                J.append(cost.numpy())
                min_SINR_vec.append(min_SINR.numpy())
                J_total.append(cost.numpy())
                min_SINR_total.append(min_SINR.numpy())
            # print(i)
            if i % 50 == 0:
                # test_rate=cost.numpy()[0]
                test_rate=np.mean(J)
#                bit2r.LR=bit2r.LR*.85
                # print('iter i=',i,'average cost is ', test_rate)
                print('Iteration = ',i,'Cost = ',np.mean(J),'sir_min_av = ',np.mean(min_SINR_vec))
                # if test_rate > best_test_rate:
                best_test_rate = test_rate
                best_W = obj.get_weights()
                save_model(obj, 'models/'+mode+'UNN_'+current_time+'.mod')

                with train_summary_writer.as_default():
                    tf.summary.scalar('test rate', test_rate, step=i)
                    tf.summary.scalar('best test rate', best_test_rate, step=i)
                
    except KeyboardInterrupt:
        pass
    
    obj.set_weights(best_W)
    return J_total,min_SINR_total

def save_model(model, fn):
    W = model.get_weights()
    with open(fn, 'wb') as f:
        pickle.dump(W, f)
        
def load_model(model, fn):
    with open(fn, 'rb') as f:
        W = pickle.load(f)
    model.set_weights(W)

######    
data=Data(Nuser)
unn=UNN(Nap,Nuser,cost_type)
J_train,min_SINR_train=train(unn,data,EPOCHS,cost_type)
#tensorboard --logdir ./logs --bind_all
#---------------------------------------------
load=0
G_batch,p_frac=data(2*batch_size,.7)
# xin=np.reshape(G_batch,[batch_size,-1])
SNR = np.power(10,P_over_noise/10)*G_batch
# xin=np.reshape(np.log(SNR),[SNR.shape[0],-1])
xin=np.log(np.diagonal(SNR,axis1=1,axis2=2))
# xin = tf.linalg.diag_part(SNR)

if load:
   unn=UNN(Nap,Nuser,cost_type) 
   cost,_ = unn(xin,SNR)
   current_dir = os.getcwd()
   path= os.path.join(current_dir,'models_trained','xUNN.mod')
#   load_model(unn, 'C:\\Users\\nikbakht\\OneDrive - Nokia\\UPF\\Codes\\UNN\\Cellular\\python\\lib\\models\\xUNN.mod')
   load_model(unn,path)
# xin=(xin-unn.Xin_av)/unn.Xin_std
xin=(xin-unn.Xin_av)/unn.Xin_std
cost,SINR,min_SINR = unn.Loss(SNR,unn.Network(xin))
print('Test cost is ',cost.numpy(),' min_SINR is ',min_SINR.numpy())
RP=Plot()
SIR_NN_clip=RP.sinr_av(SNR,unn.Network(xin),Nap,Nuser)
SIR_NN=RP.sinr_av(SNR,unn.Network(xin),Nap,Nuser,'Noclip')

SIR_frac=RP.sinr_av(SNR,tf.math.log(p_frac),Nap,Nuser)
plot=Plot()
sir_vec=[SIR_NN.numpy(),SIR_frac.numpy()]
plot.cdfplot(sir_vec)
#----------------------------------------
# unique_name=time.ctime(time.time())
# unique_name=unique_name[0:19]
sio.savemat('SIR'+'Uplink'+cost_type+'.mat',{'SIR_NN':SIR_NN.numpy(),'SIR_NN_clip':SIR_NN_clip.numpy(),'SIR_frac':SIR_frac.numpy(),
                             'J_train':J_train,'min_SINR_train':min_SINR_train, 'Nap':Nap,'Nuser':Nuser})