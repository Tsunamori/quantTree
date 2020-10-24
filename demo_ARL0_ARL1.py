"""
Author: LUCA FRITTOLI
Demo for QT-EWMA online change-detection algorithm
Contains also baseline method H-CPM
Computes empirical ARL_0 and ARL_1
"""
import numpy as np
import libquanttree as qt
import libccm as ccm
from sequential_lib import QT_EWMA, H_CPM

ARL0 = 500
tau = 50
n = 2000
N = 4096
d = 2
sKL = 2
n_exp = 1000
detection_times_qt = np.zeros((n_exp, 2))
detection_times_h = np.zeros((n_exp, 2))
for i in range(n_exp):
    # compute initial and alternative Gaussian distributions phi0 and phi1
    # CCM guarantees that sKL(phi0,phi1) = sKL
    phi0 = ccm.random_gaussian(d)
    rot, shift = ccm.compute_roto_translation(phi0, sKL)
    phi1 = ccm.rotate_and_shift_gaussian(phi0, rot, shift)
    # generate training data from phi0
    data_training = np.random.multivariate_normal(phi0[0], phi0[1], N)
    sequences = []
    # generate stationary sequence to verify false positives
    sequences.append(np.random.multivariate_normal(phi0[0], phi0[1], n))
    # generate sequence with change point tau to compute detection delay
    pre = np.random.multivariate_normal(phi0[0], phi0[1], tau)
    post = np.random.multivariate_normal(phi1[0], phi1[1], n-tau)
    sequences.append(np.concatenate((pre, post)))

    qt_ewma = QT_EWMA()
    # h_cpm = H_CPM()
    # train QuantTree histogram from training_data
    qt_ewma.train(data_training)
    # test sequences with QT_EWMA and H_CPM
    for j in range(2):
        qt_stat, qt_detectionTime, qt_detection = qt_ewma.test(sequences[j], ARL0)
        # h_stat, h_detectionTime, h_detection, h_estimatedTau = h_cpm.test(sequences[j], ARL0)
        detection_times_qt[i,j] = qt_detectionTime
        # detection_times_h[i,j] = h_detectionTime

empirical_ARL_0_qt = np.mean(detection_times_qt[:,0])
empirical_ARL_0_h = np.mean(detection_times_h[:,0])
condition_qt = detection_times_qt[:,1] > tau
condition_h = detection_times_h[:,1] > tau
ARL1_qt = np.mean(detection_times_qt[condition_qt,1] - tau)
ARL1_h = np.mean(detection_times_h[condition_h,1] - tau)

print("methods:\tQT-EWMA\tH-CPM")
print("emp. ARL_0\t{}\t{}".format(empirical_ARL_0_qt, empirical_ARL_0_h))
print("ARL_1\t\t{}\t{}".format(ARL1_qt, ARL1_h))






