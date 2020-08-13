# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:36:42 2020

@author: nikbakht
"""
#---------------------------------
import tensorflow as tf
#import socket
GPU_mode = 0
if GPU_mode:
    num_GPU =0# GPU  to use, can be 0, 2
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print('Number of GPUs available :', len(gpus))
    tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[num_GPU], mem_growth)
    print('Used GPU: {}. Memory growth: {}'.format(num_GPU, mem_growth))
#------------------------------------------
import numpy as np
from datetime import datetime
from Data import Data
# from Data_conv import Data
# from UNNdebug import UNN
from UNN_uplink import UNN
from Loss_uplink import Loss
import pickle
from Plot_results_uplink import Plot
# import matplotlib.pyplot as plt
#------------------------------------------
# tf.keras.backend.set_floatx('float64')
#train_iterations = 100
batch_size = 100
# train_per_database=100
# database_size=batch_size*train_per_database
EPOCHS =int(10)
#---------------- for values other than Nuser =12 and Nap=30, the size of environmen must be adjusted in the Data class
Nuser = 8
Nap = 20
P_over_noise = 125 # dB
cost_type = 'maxmin'
# cost_type = 'maxproduct'
#-----------------------------------------

def train(obj,Dataobj,epochs,mode):
    # TF board logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    best_test_cost = float('inf')
    best_W = None
    LR= np.logspace(-3,-4, num=epochs)
    G_batch,_= Dataobj(100*batch_size)
    SNR = np.power(10.0, P_over_noise / 10.0) * G_batch
    Crossterm = tf.math.log(tf.linalg.matmul(SNR, SNR, transpose_a=True))
    Crossterm = tf.reshape(Crossterm,[Crossterm.shape[0], -1])
    Xin = tf.math.log(tf.reduce_sum(SNR,axis=1))
    # Xin = tf.reshape(tf.math.log(SNR), [SNR.shape[0], -1])
    obj.Xin_av = np.mean(Xin,axis=0)
    obj.Xin_std = np.std(Xin,axis=0)
    obj.ct_av = np.mean(Crossterm,axis=0)
    obj.ct_std = np.std(Crossterm,axis=0)
    learning_cost = []
    try:
        for i in range(epochs):
            LR_i=LR[i ]
            optimizer = tf.keras.optimizers.Adam(LR_i)
            G_batch,_= Dataobj(100*batch_size)
            SNR = np.power(10.0, P_over_noise / 10.0) * G_batch
            crossterm = tf.math.log(tf.linalg.matmul(SNR, SNR, transpose_a=True))
            crossterm = tf.reshape(crossterm, [crossterm.shape[0], -1])
            # xin = tf.reshape(tf.math.log(SNR), [SNR.shape[0], -1])
            xin = tf.math.log(tf.reduce_sum(G_batch,axis=1))
            xin = (xin-obj.Xin_av)/obj.Xin_std
            xcrossterm = (crossterm-obj.ct_av)/obj.ct_std
            xin = tf.concat([xin, xcrossterm], axis=1)
            J=[]
            min_SINR_vec =[]
            for j in range(200):
                index = tf.random.uniform([batch_size],0,xin.shape[0],dtype=tf.dtypes.int32)
                xin_j = tf.gather(xin,index,axis=0)
                G_batch_j = tf.gather(G_batch,index,axis=0)
                with tf.GradientTape() as tape:
                    # Forward pass.
                    cost, _,min_SINR= obj(xin_j,G_batch_j)
                    # Get gradients of loss wrt the weights.
                    gradients = tape.gradient(cost, obj.trainable_weights)
                    # Gradient clipping
                    c_gradients,grad_norm = tf.clip_by_global_norm(gradients, 1.0)
                    # Update the weights of our linear layer.
                    # optimizer.apply_gradients(zip(c_gradients, obj.trainable_weights))
            grad_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(gradients[0]), 'float32')).numpy()
            if grad_nan:
                pass
            else:
                optimizer.apply_gradients(zip(gradients, obj.trainable_weights))
            J.append(cost.numpy())
            min_SINR_vec.append(min_SINR.numpy())
            learning_cost.append(cost.numpy())
            if i % 10 == 0:
                test_cost = np.mean(J)
#                bit2r.LR=bit2r.LR*.85
                print('Iteration = ', i, 'Cost = ', np.mean(J), 'sir_min_av = ', np.mean(min_SINR_vec))
                if test_cost < best_test_cost:
                    best_test_cost = test_cost
                    best_W = obj.get_weights()
                    save_model(obj, 'models/'+cost_type+'UNN.mod')

                with train_summary_writer.as_default():
                    tf.summary.scalar('test rate', test_cost, step=i)
                    tf.summary.scalar('best test rate', best_test_cost, step=i)
                
    except KeyboardInterrupt:
        pass
    
    obj.set_weights(best_W)
    return learning_cost
#-----------------------------------
def save_model(model, fn):
    W = [model.get_weights(), model.Xin_av, model.Xin_std,model.ct_av,model.ct_std]
    with open(fn, 'wb') as f:
        pickle.dump(W, f)


def load_model(model, fn):
    with open(fn, 'rb') as f:
        W = pickle.load(f)
        # model = pickle.load(f)
    model.set_weights(W[0])
    model.Xin_av = W[1]
    model.Xin_std = W[2]
    model.ct_av = W[3]
    model.ct_std = W[4]
        
# train
data = Data(Nap,Nuser)
unn = UNN(Nap,Nuser,cost_type)
learning_cost = train(unn,data,EPOCHS,'x')
#--------Create test data
G_batch,p_frac= data(200)
SNR = np.power(10.0, P_over_noise / 10.0) * G_batch
crossterm = tf.math.log(tf.linalg.matmul(SNR, SNR, transpose_a=True))
crossterm = tf.reshape(crossterm, [crossterm.shape[0], -1])
# xin = tf.reshape(tf.math.log(SNR), [SNR.shape[0], -1])
xin = tf.math.log(tf.reduce_sum(G_batch, axis=1))
xin = (xin - unn.Xin_av) / unn.Xin_std
xcrossterm = (crossterm - unn.ct_av) / unn.ct_std
xin = tf.concat([xin, xcrossterm], axis=1)


p = unn.Network(xin)
plot =Plot(Nap,Nuser)
sinr_NN = plot.sinr_averaged_fading(SNR,p)
sinr_frac = plot.sinr_averaged_fading(SNR,p_frac)
plot.cdfplot([sinr_NN.numpy(),sinr_frac.numpy()])
_,SINR_NN,_ = unn.Loss(SNR,p)
_,SINR_frac,_ = unn.Loss(SNR,p_frac)
plot.cdfplot([SINR_NN.numpy(),SINR_frac.numpy()])
