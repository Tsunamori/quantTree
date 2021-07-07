#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:16:19 2021

@author: luca
"""

import numpy as np
#import scipy.io as sio
import libquanttree as qt
import libccm as ccm
from sequential_lib import qt_online_batch, spll_online_batch, qt_ewma_test
from sequential_lib import get_qt_ewma_thresholds, get_qt_batchwise_thresholds

methods = ["QT-EWMA", "QT\t", "SPLL\t"]
n_methods = len(methods)
l = 0.03

def experiments(tr_data, sequence, ngauss, thresholds_qtewma, threshold_qt_batch, ARL_0):
    global K, nu, n_methods, do_PCA
    
    l_sequence, _ = sequence.shape
    # construct QT histogram
    qtree = qt.QuantTree(K, do_PCA)
    qtree.build_partitioning(tr_data)
    
    # fit Gaussian Mixture Model on part of training data
    gmm = qt.GaussianMixtureDataModel(ngauss)
    start = int(len(tr_data)/4)
    gmm.train_model(tr_data[:start])
    
    # bootstrap to compute threshold for SPLL on the rest of training data
    b = 0
    n = len(tr_data) - start
    thr_seq = tr_data[start:]
    H = []
    while (b+1) * nu < n-start:
        H.append(gmm.assess_goodness_of_fit(thr_seq[b*nu:(b+1)*nu]))
        b += 1
    alpha = nu / ARL_0
    threshold_spll = np.percentile(np.array(H), (1-alpha)*100)
    
    # get stopping times for QT-EWMA, QT, and SPLL
    index = np.zeros(n_methods, dtype=int)
    index[0], _ = qt_ewma_test(qtree, sequence, K, l, thresholds_qtewma)
    index[1], _ = qt_online_batch(qtree, sequence, nu, threshold_qt_batch)
    index[2], _ = spll_online_batch(gmm, sequence, nu, threshold_spll)
        
    return index

    
N = 4096
dim = 2
ARL_0 = 1000
# possible ARL_0 values: 500, 1000, 2000, 5000, 10000, 20000
target_sKL = 1
K = 32
nu = 32
do_PCA = 'none'
n_exp = 100
l_sequence = 10000
cp = 300
target_fa = 1 - (1 - 1/ARL_0)**cp

ngauss = 1

# get thresholds
thresholds_qtewma = get_qt_ewma_thresholds(l,K,ARL_0,6*ARL_0)
threshold_qt_batch = get_qt_batchwise_thresholds(nu,K,ARL_0)
detection_times = np.zeros((n_methods,n_exp), dtype=int)
stopping_times = np.zeros((n_methods,n_exp), dtype=int)


for j in range(n_exp):
    # generate a random Gaussian distribution
    gauss0 = ccm.random_gaussian(dim)
    
    # compute the alternative distribution
    rot, shift = ccm.compute_roto_translation(gauss0, target_sKL)
    gauss1 = ccm.rotate_and_shift_gaussian(gauss0, rot, shift)
    
    # generate a stationary dataset
    tr_data = np.random.multivariate_normal(gauss0[0], gauss0[1], N)
    
    # generate a sequence without changes
    sequence0 = np.random.multivariate_normal(gauss0[0], gauss0[1], 6*ARL_0)
    
    # test the sequence
    index = experiments(tr_data, sequence0, ngauss, thresholds_qtewma, threshold_qt_batch, ARL_0)
    stopping_times[:,j] = index
    
    # generate a sequence with change point at cp
    pre = np.random.multivariate_normal(gauss0[0], gauss0[1], cp)
    post = np.random.multivariate_normal(gauss1[0], gauss1[1], l_sequence-cp)
    sequence = np.concatenate((pre,post))
    
    # test the sequence
    index = experiments(tr_data, sequence, ngauss, thresholds_qtewma, threshold_qt_batch, ARL_0)
    detection_times[:,j] = index

print("method\t delay\t FA rate\t (target)\t ARL_0\t\t (target)")
for i, method in enumerate(methods):
    tp = np.where(detection_times[i,:] >= cp)
    fp = np.where(detection_times[i,:] < cp)
    avg_detection_delay = np.mean(detection_times[i,tp] - cp)
    fa_rate = len(fp[0]) / n_exp
    empirical_ARL = np.mean(stopping_times[i,:])
    print(method + "\t {:.2f}\t {:.2f}\t {:.2f}\t\t {:.2f}\t\t {:.2f}".format(avg_detection_delay,fa_rate,target_fa,empirical_ARL,ARL_0))