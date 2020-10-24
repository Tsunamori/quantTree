"""
Author: LUCA FRITTOLI
Library for QT-EWMA online change-detection algorithm
Contains also baseline method H-CPM
"""

import numpy as np
import libquanttree as qt
#import libccm as ccm
import copy

class QT_EWMA:
    def __init__(self):
        self.qtree = None

    def train(self, data_training, K=32):
        self.qtree = qt.QuantTree(K, 'pca')
        self.qtree._build_partitioning(data_training)

    def compute_stat(self, sequence, l=0.03):
        n = sequence.shape[0]
        pi_values = self.qtree.pi_values
        K = len(pi_values)
        z = pi_values
        H = np.zeros(n)
        for i in range(n):
            x = np.zeros(K)
            #x[int(self.qtree._find_bin(np.array([sequence[i]])))] = 1
            x[int(self.qtree._find_bin(sequence[i]))] = 1
            z = (1-l) * z + l * x
            stat = np.sum((z - pi_values)**2 / pi_values)
            H[i] = stat
        return H

    def test(self, sequence, ARL0, l=0.03):
        if not self.qtree.is_unif:
            print("The histogram must be uniform")
            return 0, 0
        thresholds_pol = self.get_thresholds(ARL0)
        if thresholds_pol is None:
            print("Select one of the available ARL0 values")
            return 0, 0, 0
        n = sequence.shape[0]
        pi_values = self.qtree.pi_values
        K = len(pi_values)
        z = pi_values
        H = np.zeros(n)
        change_detected = False
        i = 0
        for i in range(n):
            x = np.zeros(K)
            #print(self.qtree.find_bin(sequence[i]))
            #x[int(self.qtree._find_bin(np.array([sequence[i]])))] = 1
            x[int(self.qtree._find_bin(sequence[i]))] = 1
            
            z = (1-l) * z + l * x
            H[i] = np.sum((z - pi_values)**2 / pi_values)
            if i > 0 and H[i] > thresholds_pol(1/i):
                change_detected = True
                break
        return H, i, change_detected

    def get_thresholds(self, ARL0):
        if ARL0 == 200:
            p = np.array([613151.7261956969,
                        -1575166.5617236195,
                        1572403.091093047,
                        -815673.6559645879,
                        245354.78268305064,
                        -44644.08010174128,
                        4847.042496389174,
                        -275.4261323705265,
                        2.4633592585171082,
                        0.7263719891978682])
        elif ARL0 == 500:
            p = np.array([185587.82776714704,
                        -595672.6265179085,
                        761014.7477450933,
                        -502451.25025342824,
                        187356.24708600284,
                        -40476.24096200245,
                        4934.728249133064,
                        -296.6760376540032,
                        2.536405927307866,
                        0.8147948016391385])
        elif ARL0 == 1000:
            p = np.array([106621.6048738111,
                        -418764.549864016,
                        622846.3260776813,
                        -457287.16995146713,
                        183014.17919482998,
                        -41254.98820715151,
                        5131.298315913676,
                        -309.9870071049475,
                        2.5165375460990087,
                        0.8783070694973658])
        elif ARL0 == 2000:
            p = np.array([-308752.31602472026,
                        625781.0999577371,
                        -378156.02669484663,
                        23719.01668749218,
                        57169.612962844876,
                        -23332.61211412287,
                        3838.6816625866963,
                        -270.1789510444075,
                        1.8899969546244777,
                        0.9407942300435164])
        elif ARL0 == 5000:
            p = np.array([-352475.4257818047,
                        746326.7197034543,
                        -505609.3043336791,
                        89755.88972762412,
                        39663.54774881834,
                        -21148.138586944468,
                        3758.3527137958417,
                        -274.36614638113724,
                        1.81323879911544,
                        1.0199934297128934])
        elif ARL0 == 10000:
            p = np.array([-269768.9737019533,
                        512798.4855281412,
                        -246240.57771799687,
                        -57964.81350740981,
                        86291.6147464127,
                        -29325.712846048857,
                        4514.035514637005,
                        -307.13143196384823,
                        2.1027485134429593,
                        1.078944780099917])
        elif ARL0 == 20000:
            p = np.array([779563.495699462,
                        -1989117.793132461,
                        1969864.4583122479,
                        -1014962.113691604,
                        304559.22518984036,
                        -55677.09608759797,
                        6118.76337173538,
                        -352.38870116589436,
                        2.41965724304303,
                        1.137659410995284])
        else:
            return None
        return np.poly1d(p)

class H_CPM:
    def compute_stat(self, sequence, start=25):
        d, n = sequence.shape
        start = max(d+2, start)
        xbar_n = np.mean(sequence[:start, :], axis=0)
        T_n = np.zeros((d, d))
        for i in range(start):
            T_n += (sequence[i, :] - xbar_n).reshape((d, 1)) @ (sequence[i, :] - xbar_n).reshape((1, d))
        H = np.zeros(n)
        for i in range(start, n):
            diff = sequence[i, :] - xbar_n
            xbar_n += diff / (i + 1)
            T_n += diff.reshape((d, 1)) @ diff.reshape((1, d)) * i / (i + 1)
            T_inv = np.linalg.inv(T_n)
            offline_stat = np.zeros(i + 1)
            for k in range(1, i + 1):
                c = k * (i + 1) / (i + 1 - k)
                xbar_k = np.mean(sequence[:k], axis=0)
                diff2 = xbar_k - xbar_n
                mul = c * diff2.reshape((1, d)) @ T_inv @ diff2.reshape((d, 1))
                offline_stat[k] = (i - 1) * mul / (1 - mul)
            H[i] = max(offline_stat)
        return H

    def test(self, sequence, ARL0, start=25):
        n, d = sequence.shape
        k1, k2, k3 = self.get_params(d, ARL0)
        if k1 is None:
            print("Select one of the available ARL0 values")
            return 0, 0, 0, 0
        start = max(d + 2, start)
        xbar_n = np.mean(sequence[:start, :], axis=0)
        T_n = np.zeros((d, d))
        for i in range(start):
            T_n += (sequence[i, :] - xbar_n).reshape((d, 1)) @ (sequence[i, :] - xbar_n).reshape((1, d))
        H = np.zeros(n)
        change_detected = False
        hat_tau = 0
        for i in range(start, n):
            diff = sequence[i, :] - xbar_n
            xbar_n += diff / (i + 1)
            T_n += diff.reshape((d, 1)) @ diff.reshape((1, d)) * i / (i + 1)
            T_inv = np.linalg.inv(T_n)
            offline_stat = np.zeros(i + 1)
            for k in range(1, i + 1):
                c = k * (i + 1) / (i + 1 - k)
                xbar_k = np.mean(sequence[:k], axis=0)
                diff2 = xbar_k - xbar_n
                mul = c * diff2.reshape((1, d)) @ T_inv @ diff2.reshape((d, 1))
                offline_stat[k] = (i - 1) * mul / (1 - mul)
            H[i] = max(offline_stat)
            if H[i] > np.exp(k1 + k2 - k3 * np.log(i)):
                change_detected = True
                hat_tau = np.argmax(offline_stat)
                break
        return H, i, change_detected, hat_tau

    def get_params(self, d, ARL0):
        if ARL0 == 200:
            k1, k2, k3 = 2.706, 0.230 * d, (d + 3) / 50
        elif ARL0 == 500:
            k1, k2, k3 = 2.897, 0.226 * d, (2 * d + 7) / 100
        elif ARL0 == 1000:
            k1, k2, k3 = 3.043, 0.221 * d, (d + 4) / 50
        else:
            k1, k2, k3 = None, None, None
        return k1, k2, k3
