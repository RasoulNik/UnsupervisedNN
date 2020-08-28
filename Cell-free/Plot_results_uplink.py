# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:56:46 2020

@author: nikbakht
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Plot():
    def __init__(self,Nap,Nuser,**kwargs):
        super(Plot, self).__init__(**kwargs)
        self.Nap = Nap
        self.Nuser = Nuser
    def cdfplot(self, x):
        fig, ax = plt.subplots(1, 1)
        for i in range(len(x)):
            qe, pe = self.ecdf(10 * np.log10(x[i].flatten()))
            ax.plot(qe, pe, lw=2, label=str(i))

        # ax.hold(True)

        ax.set_xlabel('(SINR)dB')
        ax.set_ylabel('CDF')
        ax.legend(fancybox=True, loc='right')
        #    plt.xlim([-10,30])
        plt.ylim([0, 1])
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
    def sinr(self,SNR,p):
        Hloop =100
        Hreal = tf.random.normal([SNR.shape[0], self.Nap, self.Nuser, Hloop])
        Himag = tf.random.normal([SNR.shape[0], self.Nap, self.Nuser, Hloop])
        H = 1 / np.sqrt(2.0) * (tf.complex(Hreal, Himag))
        Habs = tf.math.abs(H)
        Habs = tf.cast(tf.math.real(Habs), 'float32')
        gain = tf.tile(tf.expand_dims(SNR,axis=3),[1,1,1,Hloop])* tf.math.square(Habs)
        num = tf.expand_dims(p,axis=2)*tf.square(tf.reduce_sum(gain,axis=1))

        SNR_c = tf.expand_dims(tf.complex(tf.math.sqrt(SNR),0.0),axis=3)
        gain_c_conj = tf.reshape(SNR_c*tf.math.conj(H),[H.shape[0]*Hloop,self.Nap,self.Nuser])
        gain_c = tf.reshape(SNR_c*H,[H.shape[0]*Hloop,self.Nap,self.Nuser])

        cross_gain = tf.linalg.matmul(gain_c_conj,gain_c,transpose_a=True)
        cross_gain = tf.math.real(tf.math.square(tf.math.abs(cross_gain)))
        p = tf.reshape(tf.tile(tf.expand_dims(p,axis=2),[1,1,Hloop]),[-1,self.Nuser,1])
        denom = tf.squeeze(tf.linalg.matmul(cross_gain,p))
        denom = denom -tf.linalg.diag_part(cross_gain)*tf.squeeze(p)
        denom = tf.reshape(denom,[H.shape[0],self.Nuser,Hloop])
        denom_noise = tf.reduce_sum(gain,axis=1)
        SINR = tf.reduce_mean(num/(denom+denom_noise),axis=2)
        return SINR

    def sinr_averaged_fading(self,SNR, p):
        Hloop = 100
        num = tf.zeros([SNR.shape[0], Hloop], dtype='float64')
        denom = tf.zeros(num.shape, dtype='float64')
        denomCI = tf.zeros(num.shape, dtype='float64')
        denomCH = tf.zeros(num.shape, dtype='float64')
        deonNoise = tf.zeros(num.shape, dtype='float64')
        TotalCI = tf.zeros([SNR.shape[0], self.Nuser, Hloop], dtype='float64')

        H =  1 / np.sqrt(2) * (np.random.randn(SNR.shape[0], self.Nap, self.Nuser, Hloop) +
                               1j * np.random.randn(SNR.shape[0], self.Nap, self.Nuser,Hloop))
        H = np.sqrt(np.tile(np.expand_dims(SNR / (1 + SNR), axis=3), [1, 1, 1, Hloop])) * H
        H = H.astype('complex64')
        SINR = np.zeros([SNR.shape[0], self.Nuser], dtype='float64')

        for k in range(int(self.Nuser)):
            num = (tf.tile(p[:, [k]], [1, Hloop]) * tf.square(tf.abs(np.sum(
                tf.tile(SNR[:, :, [k]], [1, 1, Hloop]) * np.square(np.abs(H[:, :, k, :])), axis=1))))

            TotalCI = (tf.tile(tf.expand_dims(p, axis=2), [1, 1, Hloop]) * np.square(
                np.abs(np.sum(np.tile(np.conj(H[:, :, [k], :]), [1, 1, self.Nuser, 1]) *
                              np.tile(np.sqrt(tf.expand_dims(SNR[:, :, [k]], axis=3)), [1, 1, self.Nuser, Hloop]) * (
                                  H) * np.tile(np.sqrt(tf.expand_dims(SNR, axis=3)), [1, 1, Hloop])
                              , axis=1))))

            denomCI = np.sum(TotalCI, axis=1) - TotalCI[:, k, :]

            denomCH = tf.reduce_sum(
                tf.multiply(tf.multiply(tf.tile(SNR[:, :, [k]], [1, 1, Hloop]), tf.square(tf.abs(H[:, :, k, :]))),
                            tf.tile(tf.reduce_sum(
                                tf.multiply(tf.tile(tf.expand_dims(p, axis=1), [1, self.Nap, 1]), tf.divide(SNR, 1 + SNR)),
                                axis=2, keepdims=True), [1, 1, Hloop]))
                , axis=1)
            denomNoise = tf.reduce_sum(
                tf.multiply(tf.tile(SNR[:, :, [k]], [1, 1, Hloop]), tf.square(tf.abs(H[:, :, k, :]))), axis=1)

            denom = denomCI + denomCH + denomNoise
            SINR[:, [k]] = np.expand_dims(np.mean(num / denom, axis=1), axis=1)

        #    Cost=Cost+(tf.reduce_sum(tf.exp(-10*p),axis=1))
        #    Cost=tf.reduce_sum(tf.exp(-1*p),axis=1)
        #    +100*tf.square(tf.norm(p,ord='euclidean',axis=1)-1)
        return SINR

