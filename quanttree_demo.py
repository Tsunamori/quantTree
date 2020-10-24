import numpy as np
import libquanttree as qt
import libccm as ccm

from libquanttree import DataModel, ThresholdStrategy


class ChangeDetectionTest:

    def __init__(self, model: DataModel, nu: int, threshold_strat: ThresholdStrategy):
        self.model = model
        self.threshold_strat = threshold_strat
        self.nu = nu

    def train(self, _data):
        self.model.train_model(_data)
        self.threshold_strat.configure_strategy(self.model, _data)

    def reject_null_hypothesis(self, _data, _alpha):
        _stat = self.compute_statistic(_data)
        _threshold = self.threshold_strat.get_threshold(_alpha)
        return _stat > _threshold, _stat

    def compute_statistic(self, _data):
        _stat = self.model.assess_goodness_of_fit(_data)
        return _stat


# --- Demo parameters

dim = 8  # data dimension
K = 64  # number of bins of the histogram
N = 4096  # number of samples in the training set
nu = 128  # number of samples in the test windws
alpha = 0.05  # signifance of the test

do_PCA = True  # rotate data to align them to coordinate axes
target_sKL = 1  # target value of the symmetric kullback-leibler
nbatches = 1000  # Number of batches for testing

if do_PCA:
    transf_type = 'pca'
else:
    transf_type = 'none'

# --- Generating the distributions

# Generate a random gaussian distribution
gauss0 = ccm.random_gaussian(dim)

# Generate a random roto-translation yielding a changed desitribution with the desired Kullback-Leibler divergence
rot, shift = ccm.compute_roto_translation(gauss0, target_sKL)

# Compute the alternative distribution
gauss1 = ccm.rotate_and_shift_gaussian(gauss0, rot, shift)

# --- Generating training data

# Generate stationary data
data = np.random.multivariate_normal(gauss0[0], gauss0[1], N)

# --- Building a QuantTree and preparing the Change Detection Test

# QuantTree (partitioning)
qtree = qt.QuantTree(pi_values=K, transformation_type=transf_type)

# Histogram (data model)
hist = qt.Histogram(partitioning=qtree, statistic_name='pearson')

# QuantTreeThresholdStrategy (computes the threshold as it is described in the paper)
th_strat = qt.QuantTreeThresholdStrategy(nu=nu, ndata_training=N, pi_values=K, statistic_name='pearson')

# ChangeDetectionTest
cdt = ChangeDetectionTest(model=hist, nu=nu, threshold_strat=th_strat)

# Training the ChangeDetectionTest object does the following:
#  a) builds the QuantTree (qtree.build_partitioning)
#  b) computes pi values from data (hist.estimate_probabilities)
#  c) stores all the info in the ChangeDetectionTest object
cdt.train(data)

# --- Threshold computation and testing

# Compute the threshold
threshold = cdt.threshold_strat.get_threshold(alpha)
print("Alpha = {} - Threshold = {}".format(alpha, threshold))

# Data ~ normal distribution
normal_detected = np.zeros(nbatches)
normal_stats = np.zeros(nbatches)
for i in range(nbatches):
    batch = np.random.multivariate_normal(gauss0[0], gauss0[1], nu)
    normal_detected[i], normal_stats[i] = cdt.reject_null_hypothesis(batch, alpha)

print("FPR: {}".format(normal_detected.sum() / normal_stats.size))

# Data ~ change distribution
change_detected = np.zeros(nbatches)
change_stats = np.zeros(nbatches)
for i in range(nbatches):
    batch = np.random.multivariate_normal(gauss1[0], gauss1[1], nu)
    change_detected[i], change_stats[i] = cdt.reject_null_hypothesis(batch, alpha)

print("TPR: {}".format(change_detected.sum() / change_detected.size))
