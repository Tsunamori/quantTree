import numpy as np

def PearsonStat(y,p,nu):
    return np.sum(y**2/p) / nu - nu

def qt_ewma_stat(qtree, sequence, K, l):
    n = len(sequence)
    z = np.ones(K) / K
    pi = 1 / K
    H = []
    for i in range(n):
        x = np.zeros(K)
        J = int(qtree.find_bin(np.array([sequence[i]])))
        x[J] = 1
        z = (1-l) * z + l * x
        stat = np.sum((z - pi)**2 / pi)
        H.append(stat)
    return H

def qt_ewma_test(qtree, sequence, K, l, thresholds):
    n = len(sequence)
    z = np.ones(K) / K
    pi = 1 / K
    i = 0
    change_detected = False
    while i < n and change_detected == False:
        x = np.zeros(K)
        J = int(qtree.find_bin(np.array([sequence[i]])))
        x[J] = 1
        z = (1-l) * z + l * x
        stat = np.sum((z - pi)**2 / pi)
        if stat > thresholds[i]:
            change_detected = True
        else:
            i += 1
    return i, change_detected

def qt_online_batch(qtree, sequence, nu, h):
    n = len(sequence)
    K = len(qtree.pi_values)
    seq_bin = qtree.find_bin(sequence).astype(int)
    Y = np.zeros((K,n), dtype=int)
    Y[seq_bin, np.arange(n)] = 1
    p = np.ones(K) / K
    b = 0
    t = b * nu
    change_detected = False
    H = []
    while (b+1) * nu < n and change_detected == False:
        stat = PearsonStat(np.sum(Y[:,b*nu:(b+1)*nu], axis=1), p, nu)
        H.append(stat)
        if stat > h:
            change_detected = True
        b += 1
        t = b * nu
    return t, change_detected

def spll_online_batch(gmm, sequence, nu, h):
    n = len(sequence)
    b = 0
    t = b * nu
    change_detected = False
    H = []
    while (b+1) * nu < n and change_detected == False:
        stat = gmm.assess_goodness_of_fit(sequence[b*nu:(b+1)*nu])
        H.append(stat)
        if stat > h:
            change_detected = True
        b += 1
        t = b * nu
    return t, change_detected

def get_poly(ARL0):
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

def get_qt_ewma_thresholds(l,K,ARL_0,l_sequence):
    poly = get_poly(ARL_0)
    
    thresholds = np.zeros(l_sequence)
    thresholds[0] = 1000
    for ii in range(1,l_sequence):
        thresholds[ii] = poly(1/ii)
    return thresholds

def get_qt_batchwise_thresholds(nu,K,ARL_0):
    lookup = np.load("qt_batchwise_stats_{}_{}.npz".format(K,nu))['arr_0']
    alpha = nu / ARL_0
    threshold = np.percentile(lookup, (1-alpha)*100)
    return threshold
