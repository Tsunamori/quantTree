"""
Author: LUCA FRITTOLI
Demo for QT-EWMA online change-detection algorithm
Contains also baseline method H-CPM
"""
import numpy as np
import libquanttree as qt
import libccm as ccm
from sequential_lib import QT_EWMA, H_CPM

ARL0 = 1000
tau = 50
n = 2000
N = 4096
d = 2
sKL = 2
# compute initial and alternative Gaussian distributions phi0 and phi1
# CCM guarantees that sKL(phi0,phi1) = sKL
phi0 = ccm.random_gaussian(d)
rot, shift = ccm.compute_roto_translation(phi0, sKL)
phi1 = ccm.rotate_and_shift_gaussian(phi0, rot, shift)
# generate training data from phi0
data_training = np.random.multivariate_normal(phi0[0], phi0[1], N)
# generate sequence with change point tau
pre = np.random.multivariate_normal(phi0[0], phi0[1], tau)
post = np.random.multivariate_normal(phi1[0], phi1[1], n-tau)
sequence = np.concatenate((pre, post))

qt_ewma = QT_EWMA()
h_cpm = H_CPM()
# train QuantTree histogram from training_data
qt_ewma.train(data_training)
# test sequence with QT_EWMA and H_CPM
qt_stat, qt_detectionTime, qt_detection = qt_ewma.test(sequence, ARL0)
h_stat, h_detectionTime, h_detection, h_estimatedTau = h_cpm.test(sequence, ARL0)

print("Change point at tau = {}".format(tau))
print("QT-EWMA went off at t* = {}".format(qt_detectionTime))
print("H-CPM went off at t* = {}".format(h_detectionTime))


