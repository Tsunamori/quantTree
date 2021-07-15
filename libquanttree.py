import numpy as np
import scipy.stats
import libccm as ccm
import time
from abc import ABC, abstractmethod
import pickle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import mvn
import scipy.stats
from itertools import product


# Function to give data the proper shape
def reshape_data(data) -> np.ndarray:
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    return data


# Total variation statistic
def tv_statistic_(_pi_exp, _pi_hats, _nu):
    # _pi_exp.shape = (K,) or (nbatches, K)
    # _pi_hats.shape == (nbatches, K)
    _stat = _nu * np.abs(_pi_exp - _pi_hats)
    _axis = len(_stat.shape) - 1
    return 0.5 * np.sum(_stat, axis=_axis)


# Pearson statistic
def pearson_statistic_(_pi_exp, _pi_hats, _nu):
    # _pi_exp.shape = (K,) or (nbatches, K)
    # _pi_hats.shape == (nbatches, K)
    _stat = _nu * ((_pi_exp + - _pi_hats) ** 2) / _pi_exp
    _axis = len(_stat.shape) - 1
    return np.sum(_stat, axis=_axis)


# Abstract class representing a transformation of the data.
# The subclasses has to implement two methods:
# - estimate_transformation: estimates the parameter of the transformation if necessary, e.g. in case of PCARotation.
#       It takes as input the training data
# - transform_data: it actually transforms the input data and returns the transformed data
# It has a concrete factory method for each implemented subclasses, i.e.:
# - Identity: it does not perform any transformation of data
# - RandomRotation: it rotates the data w.r.t. a random rotation matrix generated at initialization time
# - PCARotation: based on the PCA estimated from training data
class DataTransformation(ABC):

    @abstractmethod
    def estimate_transformation(self, data: np.ndarray):
        pass

    @abstractmethod
    def transform_data(self, data: np.ndarray):
        pass

    @classmethod
    def get_data_transformation(cls, transformation_type: str = 'none'):
        if str.lower(transformation_type) == 'none':
            return Identity()
        if str.lower(transformation_type) == 'pca':
            return PCARotation()
        if str.lower(transformation_type) == 'random':
            return RandomRotation()


# Abstract class representing a partitioning of data space. Each subclass has to implement two (protected) methods:
# - _build_partitioning: given a training set, it builds the partitioning og the data space
# - _find_bin: given a batch of data it returns the indices of the bin for each data
# The constructor of Partitioning class takes as input a DataTransformation object. The Partitioning class transforms
# training and test data according to this transformation (the methods _find_bin and _build_partitioning DO NOT have to
# implement the tranformation of the data)
class Partitioning(ABC):

    def __init__(self, nbin: int = 2, transformation_type: str = 'none'):
        self.nbin: int = nbin
        self.transformation: DataTransformation = DataTransformation.get_data_transformation(transformation_type)

    def find_bin(self, data):
        data = self.transform_data(data)
        return self._find_bin(data)

    @abstractmethod
    def _find_bin(self, data):
        pass

    def get_bin_counts(self, data):
        data = self.transform_data(data)
        return self._get_bin_counts(data)

    @abstractmethod
    def _get_bin_counts(self, data):
        pass

    def build_partitioning(self, data):
        data = reshape_data(data)
        self.transformation.estimate_transformation(data)
        data = self.transform_data(data)
        self._build_partitioning(data)

    @abstractmethod
    def _build_partitioning(self, data):
        pass

    def transform_data(self, data):
        return self.transformation.transform_data(data)

    @abstractmethod
    def compute_gaussian_probabilities(self, gauss):
        pass

    def jaccard_distance(self, partitioning, data: np.ndarray):
        idx_bin_1 = self.find_bin(data)
        idx_bin_2 = partitioning.find_bin(data)

        all_bin_1 = [set([i_data for i_data, idx in enumerate(idx_bin_1) if idx == ii]) for ii in range(self.nbin)]
        all_bin_2 = [set([i_data for i_data, idx in enumerate(idx_bin_2) if idx == ii]) for ii in range(self.nbin)]

        all_dist = np.zeros((self.nbin, partitioning.nbin))
        for ii in range(self.nbin):
            for jj in range(partitioning.nbin):
                all_dist[ii, jj] = 1 - len(all_bin_1[ii].intersection(all_bin_2[jj])) / len(
                    all_bin_1[ii].union(all_bin_2[jj]))

        return 0.5 * (np.mean(np.min(all_dist, axis=0)) + np.mean(np.min(all_dist, axis=1)))


class Statistic:

    def __init__(self, statistic_name: str = 'pearson', pi_values=None):
        if statistic_name == 'pearson':
            self.statistic = pearson_statistic_
        elif statistic_name == 'tv':
            self.statistic = tv_statistic_
        else:
            raise Exception("Invalid statistic name: {}".format(statistic_name))

        self.pi_values = pi_values

    def compute_statistic(self, pi_hats, nu):
        return self.statistic(_pi_exp=self.pi_values, _pi_hats=pi_hats, _nu=nu)


# Abstract class representing a model for data. It has two methods: one for training (train_model) and one to assess the
# goodness of fit of test data to the learned model
class DataModel(ABC):

    @abstractmethod
    def assess_goodness_of_fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def train_model(self, data: np.ndarray):
        pass


# Class representing a histogram as data model. The histogram is defined through a Partitioning object, while the
# goodness of fit is assess through the Pearson statistic or total variation distance (the choice among the two is made
# at contruction time.
class Histogram(DataModel):

    def __init__(self, partitioning: Partitioning, statistic_name: str = 'pearson'):
        # Underlying space partitioning
        self.partitioning: Partitioning = partitioning

        # Statistic name and instance (initialized as None)
        self.statistic_name: str = statistic_name
        self.statistic: Statistic = Statistic()

        # Uniformly initialized pi_values
        self.pi_values: np.ndarray = np.ones(self.partitioning.nbin) * 1 / self.partitioning.nbin

    def __hash__(self):
        tmp = (self.statistic_name.__hash__(), self.partitioning.__hash__(), tuple(self.pi_values).__hash__())
        return tmp.__hash__()

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.statistic_name == other.statistic_name and self.partitioning == other.partitioning and tuple(
                self.pi_values) == tuple(other.pi_values)

    def assess_goodness_of_fit(self, data):
        # data: (N,) -> (N, d) -> (M, N, d)
        # returns (M,)
        data = reshape_data(data)
        if len(data.shape) == 2:
            data = data.reshape((1,) + data.shape)
        _, nu, _ = data.shape
        pi_hats = self.partitioning.get_bin_counts(data) / nu
        return self.statistic.compute_statistic(pi_hats=pi_hats, nu=nu)

    def train_model(self, data):
        self.partitioning.build_partitioning(data)
        self.estimate_probabilities(data)
        self.statistic = Statistic(statistic_name=self.statistic_name, pi_values=self.pi_values)

    def estimate_probabilities(self, data: np.ndarray):
        data = reshape_data(data)
        self.pi_values = self.get_bin_counts(data) / data.shape[0]

    def get_bin_counts(self, data: np.ndarray):
        # data: (M, N, d)
        # returns (M, K)
        data = reshape_data(data)
        return self.partitioning.get_bin_counts(data)


class ThresholdStrategy(ABC):

    def __init__(self):
        self.threshold_dict = dict()

    def get_threshold(self, alpha):
        threshold = self.threshold_dict.get(alpha)
        if not threshold:
            threshold = self._get_threshold(alpha)
            self.add_threshold(alpha, threshold)
        return threshold

    def add_threshold(self, alpha, threshold):
        self.threshold_dict[alpha] = threshold

    @abstractmethod
    def _get_threshold(self, alpha):
        pass

    # non necessario da implementare, ma solo se serve configurarla su un modello
    def configure_strategy(self, model: DataModel, data_training: np.ndarray):
        pass


# CLASSI CONCRETE

# DataTransformation
class RandomRotation(DataTransformation):

    def __init__(self):
        super().__init__()
        self.rotation = []

    def estimate_transformation(self, data: np.ndarray):
        # data: (N, dim)
        data = reshape_data(data)
        dim = data.shape[1]
        self.rotation = ccm.generate_random_rotation(dim)

    def transform_data(self, data):
        data = reshape_data(data)
        data_new = np.dot(data, self.rotation)
        return data_new


class PCARotation(DataTransformation):

    def __init__(self):
        super().__init__()
        self.pc = self.pc = PCA(whiten=False)

    def estimate_transformation(self, data: np.ndarray):
        data = reshape_data(data)

        if len(data) == 3:
            data = data.reshape((-1, data.shape[2]))

        self.pc.fit(data)

    def transform_data(self, data):
        data = reshape_data(data)

        if len(data.shape) == 2:
            return self.pc.transform(data)
        elif len(data.shape) == 3:
            d0, d1, d2 = data.shape
            data = data.reshape((-1, d2))
            return self.pc.transform(data).reshape((d0, d1, d2))


class Identity(DataTransformation):

    def __init__(self):
        super().__init__()

    def estimate_transformation(self, data: np.ndarray):
        pass

    def transform_data(self, data):
        return data


# Partitioning
class QuantTree(Partitioning):

    def __init__(self, pi_values, transformation_type: str = 'none'):
        pi_values = np.array(pi_values)
        if pi_values.size == 1:
            nbin = pi_values
            self.pi_values = np.ones(nbin) / nbin
            self.is_unif = True
        else:
            self.pi_values = np.array(pi_values)
            if len(np.unique(pi_values)) == 1:
                self.is_unif = True
            else:
                self.is_unif = False
            nbin = len(pi_values)
        super().__init__(nbin, transformation_type)
        self.leaves: np.ndarray = np.array([])  # MOD
        self.ndata_training = []
        self.dim = None

    def _build_partitioning(self, data):
        data = reshape_data(data)
        ndata = data.shape[0]
        ndim = data.shape[1]
        nbin = self.pi_values.size
        self.ndata_training = ndata
        self.dim = ndim

        # Each leaf is characterized by 3 numbers:
        # 1) the dimension of the split that genrates the leaf,
        # 2) the lower bound of the leaf,
        # 3) the upper bound of the leaf
        self.leaves = np.ones(shape=(3, nbin))

        # set the limits of the available space in each dimension
        limits = np.ones((2, ndim))
        limits[0, :] = -np.inf
        limits[1, :] = np.inf

        # all samples are available
        available = [True] * ndata

        # iteratively generate the leaves
        for i_leaf in range(nbin - 1):
            # select a random components
            i_dim = np.random.randint(ndim)
            x_tilde = data[available, i_dim]

            # find the indices of the available samples
            idx = [i for i in range(len(available)) if available[i]]
            N_tilde = len(idx)

            # sort the samples
            idx_sorted = sorted(range(len(x_tilde)), key=x_tilde.__getitem__)
            x_tilde.sort()

            # compute p_tilde
            p_tilde = self.pi_values[i_leaf] / (1 - np.sum(self.pi_values[0:i_leaf]))
            L = int(np.round(p_tilde * N_tilde))

            # define the leaf
            if np.random.choice([True, False]):
                self.leaves[:, i_leaf] = [i_dim, limits[0, i_dim], x_tilde[L - 1]]
                limits[0, i_dim] = x_tilde[L - 1]
                idx_sorted = idx_sorted[0:L]
            else:
                self.leaves[:, i_leaf] = [i_dim, x_tilde[-L - 1], limits[1, i_dim]]
                limits[1, i_dim] = x_tilde[-L - 1]
                idx_sorted = idx_sorted[-L:]

            # remove the sample in the leaf from the available samples
            for i in idx_sorted:
                available[idx[i]] = False

        # define the last leaf with the remaining samples
        i_dim = np.random.randint(ndim)
        self.leaves[:, -1] = [i_dim, limits[0, i_dim], limits[1, i_dim]]

    def _find_bin(self, data):
        # Expected data shape: (nbatches, nu, dim)

        # If the batch is made of only one sample, we need to reshape
        if len(data.shape) == 1:
            data = data.reshape((1, 1) + data.shape)

        # If nbatches is missing, add it through reshape
        if len(data.shape) == 1:
            data = data.reshape((1, 1) + data.shape)
        if len(data.shape) == 2:
            data = data.reshape((1,) + data.shape)

        nbatches, nu, dim = data.shape
        nbin = self.nbin

        # leaf contains the bin indexes where the data points fall
        leaf = np.ones((nbatches, nu)) * (nbin - 1)

        available = np.ones((nbatches, nu), dtype=bool)

        for i_leaf in range(nbin - 1):
            ldim = self.leaves[0, i_leaf].astype(int)

            # Get the matrix coordinates of the available input data points
            av_idx = np.array([c for c in product(range(nbatches), range(nu)) if available[c]])

            # Project available data on the current dimension
            x_tilde = data[np.where(available)][:, ldim]

            # Get the indexes of the points falling in bin[i_leaf]
            fb_idx = av_idx[np.array((self.leaves[1, i_leaf] < x_tilde) & (x_tilde < self.leaves[2, i_leaf]))]

            # Update
            leaf[fb_idx[:, 0], fb_idx[:, 1]] = i_leaf
            available[fb_idx[:, 0], fb_idx[:, 1]] = False
            if available.sum() == 0:
                break

        # the squeeze function is to ensure that 2-dimensional inputs have 2-dimensional outputs
        return leaf.squeeze()

    def _get_bin_counts(self, data):
        # data: ([nbathces], nu, dim)
        # bins: ([nbatches], nu, dim)
        # bin_counts: ([nbatches],)
        bins = self._find_bin(data)
        if len(bins.shape) == 1:
            bins = bins.reshape((1,) + bins.shape)
        bin_counts = np.zeros((bins.shape[0], self.nbin))
        for i in range(self.nbin):
            bin_counts[:, i] = np.count_nonzero(bins == i, axis=1)
        return bin_counts.squeeze()

    def get_leaves_box(self):
        nleaves = self.leaves.shape[1]
        dim = self.dim
        box = np.ndarray(shape=(2, dim, nleaves))

        box[0, :, :] = -np.inf
        box[1, :, :] = np.inf

        for i_leaf in range(nleaves):
            dim_split = int(self.leaves[0, i_leaf])
            # controllo se lo split e' stato fatto prendendo la coda destra o sinistra
            # trans: check if the split was done by taking the left or right tail
            is_lower_split = box[0, dim_split, i_leaf] == self.leaves[1, i_leaf]

            # calcolo il box
            # trans: calculate the box
            box[0, dim_split, i_leaf] = self.leaves[1, i_leaf]
            box[1, dim_split, i_leaf] = self.leaves[2, i_leaf]

            # aggiorno i limiti dei box successivi
            # trans: update the limits of the following boxes
            if is_lower_split:
                box[0, dim_split, i_leaf + 1:] = box[1, dim_split, i_leaf]
            else:
                box[1, dim_split, i_leaf + 1:] = box[0, dim_split, i_leaf]

        return box

    def compute_gaussian_probabilities(self, gauss):
        box = self.get_leaves_box()
        p0 = np.zeros(self.nbin)
        for i_K in range(self.nbin):
            lower = np.squeeze(box[0, :, i_K])
            upper = np.squeeze(box[1, :, i_K])
            value, _ = mvn.mvnun(lower, upper, gauss[0], gauss[1])
            p0[i_K] = value
        return p0


class QuantTreeUnivariate(QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)

    def _build_partitioning(self, data):
        data = np.array(data).squeeze()

        ndata = len(data)
        self.ndata_training = ndata
        L_values = np.round(self.pi_values * ndata)
        L_acc = np.cumsum(L_values)
        L_acc = [np.int(i - 1) for i in L_acc]

        x = np.sort(data)

        self.leaves = np.concatenate(([-np.inf], x[L_acc[:-1]], [np.inf]))

    def _find_bin(self, data):
        # OLD
        # data = np.array(data).squeeze()
        # if len(data.shape) == 0:
        #     data = [data]
        # leaf = np.array([np.sum(x > self.leaves) for x in data]) - 1

        # NEW
        data = np.array(data).squeeze()
        if len(data.shape) == 0:
            data = [data]
        data = data.reshape(-1, 1)
        leaf = np.sum(data > self.leaves, axis=1) - 1

        return leaf


class ParametricGaussianModel(DataModel):
    """Modello che implementa il two sample t-test per dati gaussiani multivariati"""
    # trans: Model that implements the two sample t-test for multivariate Gaussian data

    def __init__(self):
        self.mu: np.ndarray = None
        self.Sigma: np.ndarray = None
        self.ntrain_data: int = None

    def train_model(self, data: np.ndarray):
        data = reshape_data(data)
        self.mu = np.mean(data, 0)
        self.Sigma = np.cov(np.transpose(data))
        self.ntrain_data = data.shape[0]

    def assess_goodness_of_fit(self, data: np.ndarray):
        data = reshape_data(data)
        mu0 = self.mu
        Sigma0 = self.Sigma
        n0 = self.ntrain_data
        mu1 = np.mean(data, 0)
        Sigma1 = np.cov(np.transpose(data))
        n1 = data.shape[0]
        Sigma_pool = (n0 - 1) / (n0 + n1 - 2) * Sigma0 + (n1 - 1) / (n0 + n1 - 2) * Sigma1
        Sigma_pool = (1 / n0 + 1 / n1) * Sigma_pool
        t = np.dot(mu0 - mu1, np.linalg.solve(Sigma_pool, mu0 - mu1))
        return t


# ThresholdStrategy
class BootstrapThresholdStrategy(ThresholdStrategy):

    def __init__(self, model: DataModel, nu: int, nbatch: int, data: np.ndarray):
        super().__init__()
        self.model = model
        self.nu = nu
        self.nbatch = nbatch
        self.data = reshape_data(data)

    def _get_threshold(self, alpha):
        ndata = self.data.shape[0]

        stats = []
        for i_batch in range(self.nbatch):
            batch = self.data[np.random.choice(ndata, self.nu, replace=True), :]
            stats.append(self.model.assess_goodness_of_fit(batch))

        stats.sort()
        stats.insert(0, stats[0] - 1)
        threshold = stats[np.int(np.floor((1 - alpha) * self.nbatch))]
        return threshold

    # non necessario da implementare, ma solo se serve configurarla su un modello
    # trans: not necessary to implement, but only if you need to configure it on a model
    def configure_strategy(self, model: DataModel, data: np.ndarray):
        self.model = model
        self.data = reshape_data(data)


########################################################################################################################
# """
class QuantTreeThresholdStrategy(ThresholdStrategy):
    with open('all_distr_quanttree.p', 'rb') as distr_file:
        all_distr = pickle.load(distr_file)

    def __init__(self, nu: int = 1, ndata_training: int = 1, pi_values=2, statistic_name: str = 'pearson'):
        super().__init__()
        self.nu = nu
        self.N = ndata_training
        self.statistic_name = statistic_name
        pi_values = np.array(pi_values)
        if pi_values.size == 1:
            nbin = pi_values
            self.pi_values = np.ones(nbin) / nbin
            self.is_unif = True
        else:
            self.pi_values = pi_values
            if len(np.unique(pi_values)) == 1:
                self.is_unif = True
            else:
                self.is_unif = False

    @classmethod
    def get_precomputed_uniform_quanttree_threshold(cls, stat_name, alpha, nbin, ndata, nu, ):
        # threshold = {('pearson', 0.001,  32,  4096,  64): 64,
        #              ('pearson', 0.001, 128,  4096,  64): 192,
        #              (     'tv', 0.001,  32,  4096,  64): 25,
        #              (     'tv', 0.001, 128,  4096,  64): 43,
        #              ('pearson', 0.001,  32, 16384, 256): 62.75,
        #              ('pearson', 0.001, 128, 16384, 256): 187,
        #              (     'tv', 0.001,  32, 16384, 256): 52,
        #              (     'tv', 0.001, 128, 16384, 256): 85,
        #              ('pearson',  0.01,  32,  4096,  64): 54,
        #              ('pearson',  0.01, 128,  4096,  64): 172,
        #              (     'tv',  0.01,  32,  4096,  64): 23,
        #              (     'tv',  0.01, 128,  4096,  64): 42,
        #              ('pearson',  0.01,  32, 16384, 256): 53.25,
        #              ('pearson',  0.01, 128, 16384, 256): 171,
        #              (     'tv',  0.01,  32, 16384, 256): 47,
        #              (     'tv',  0.01, 128, 16384, 256): 81,
        #              ('pearson',  0.05,  32,  4096,  64): 46,
        #              ('pearson',  0.05, 128,  4096,  64): 156,
        #              (     'tv',  0.05,  32,  4096,  64): 21,
        #              (     'tv',  0.05, 128,  4096,  64): 41,
        #              ('pearson',  0.05,  32, 16384, 256): 45.75,
        #              ('pearson',  0.05, 128, 16384, 256): 157,
        #              (     'tv',  0.05,  32, 16384, 256): 44,
        #              (     'tv',  0.05, 128, 16384, 256): 78}

        saved = cls.all_distr.get((stat_name, nbin, ndata, nu))
        if not saved:
            return None
        else:
            values, ecdf = saved
            tau = values[np.sum(ecdf < 1 - alpha)]
            return tau

    def compute_distribution_quanttree(self, nbatch):
        partitioning = QuantTreeUnivariate(self.pi_values)
        K = len(self.pi_values)
        N = self.N
        nu = self.nu
        y = self.pi_values * nu
        stats = np.zeros(nbatch)
        for i_batch in range(nbatch):
            if i_batch % 1000 == 0:
                print(i_batch)
            data = np.random.uniform(0, 1, N)
            batch = np.random.uniform(0, 1, nu)
            # API standard
            # histogram.train_model(data)
            # stats[i_batch] = histogram.assess_goodness_of_fit(batch)

            # API non standard
            # partitioning._build_partitioning(data)
            # idx = partitioning._find_bin(batch)
            partitioning.build_partitioning(data)
            idx = partitioning.find_bin(batch)
            y_hat, _ = np.histogram(idx, bins=K, range=(0, K - 1))

            if self.statistic_name == 'pearson':
                stats[i_batch] = np.sum(np.abs(y - y_hat) ** 2 / y)
            elif self.statistic_name == 'tv':
                stats[i_batch] = 0.5 * np.sum(np.abs(y - y_hat))
            else:
                ValueError('Statistic not supported')

        values, counts = np.unique(stats, return_counts=True)

        ecdf = np.cumsum(counts) / nbatch

        # per ottenere il piu' piccolo threshold gamma tale per cui P(T > gamma) <= alpha basta fare
        # trans: to get the smallest threshold gamma such that P (T> gamma) <= alpha just do
        # values[np.sum(ecdf < 1-alpha)]

        return values, ecdf

    def estimate_quanttree_threshold(self, alpha, nbatch=1000):
        if not isinstance(alpha, list):
            alpha_values = [alpha]
        else:
            alpha_values = alpha

        partitioning = QuantTreeUnivariate(self.pi_values)
        K = len(self.pi_values)
        N = self.N
        nu = self.nu
        y = self.pi_values * nu
        # API standard
        # histogram = Histogram(partitioning, self.statistic_name)

        # non utilizzo le API standard perche' sono troppo lente e in questo caso vogliamo essere molto piu' rapidi
        # trans: I don't use the standard API because they are too slow and in this case we want to be much faster
        stats = np.zeros(nbatch)
        for i_batch in range(nbatch):
            if i_batch % 100 == 0:
                print(i_batch)
            data = np.random.uniform(0, 1, N)
            batch = np.random.uniform(0, 1, nu)
            # API standard
            # histogram.train_model(data)
            # stats[i_batch] = histogram.assess_goodness_of_fit(batch)

            # API non standard
            # partitioning._build_partitioning(data)
            # idx = partitioning._find_bin(batch)
            partitioning.build_partitioning(data)
            idx = partitioning.find_bin(batch)
            y_hat, _ = np.histogram(idx, bins=K, range=(0, K - 1))
            stats[i_batch] = np.sum(np.abs(y - y_hat) ** 2 / y)

        stats.sort()
        threshold_values = [stats[np.int(np.ceil((1 - alpha) * nbatch))] for alpha in alpha_values]

        if len(alpha_values) == 1:
            threshold_values = threshold_values[0]
        return threshold_values

    def _get_threshold(self, alpha):
        threshold = None
        if self.is_unif:
            threshold = QuantTreeThresholdStrategy.get_precomputed_uniform_quanttree_threshold(self.statistic_name,
                                                                                               alpha,
                                                                                               len(self.pi_values),
                                                                                               self.N, self.nu)

        if not threshold:
            print('Threshold not found, it has to be estimated')
            threshold = self.estimate_quanttree_threshold([alpha])

        return threshold

    # histogram deve essere basato su quanttree
    # trans: histogram must be quanttree based
    def configure_strategy(self, histogram: Histogram, data: np.ndarray):
        if not isinstance(histogram.partitioning, QuantTree):
            ValueError('This strategy can be used only on histograms computed using QuantTree Partitioning')
        self.pi_values = histogram.pi_values
        self.is_unif = histogram.partitioning.is_unif
        self.N = histogram.partitioning.ndata_training


# """


########################################################################################################################

class TwoSampleHotellingThresholdStrategy(ThresholdStrategy):

    def __init__(self, nu: int, ntrain_data: int = None, dim: int = None):
        super().__init__()
        self.n0 = ntrain_data
        self.n1 = nu
        self.dim = dim

    def _get_threshold(self, alpha):
        dof1 = self.dim
        dof2 = self.n0 + self.n1 + - self.dim - 1
        factor = (self.n0 + self.n1 - 2) * self.dim / (self.n0 + self.n1 + - self.dim - 1)
        return factor * scipy.stats.f.ppf(1 - alpha, dof1, dof2)

    def configure_strategy(self, model: ParametricGaussianModel, data: np.ndarray):
        data = reshape_data(data)
        self.n0 = model.ntrain_data
        self.dim = data.shape[1]


class GaussianMixtureDataModel(DataModel):

    def __init__(self, ngauss):
        self.ngauss = ngauss
        self.model = GaussianMixture(ngauss) # covariance_type='tied' is outdated
        self.cov_inv = None

    def train_model(self, data: np.ndarray):
        data = reshape_data(data)
        self.model.fit(data)
        self.cov_inv = np.linalg.inv(self.model.covariances_)

    def assess_goodness_of_fit(self, data: np.ndarray):
        data = reshape_data(data)

        all_loglike = np.zeros((data.shape[0], self.ngauss))
        for ii in range(data.shape[0]):
            v = data[ii][np.newaxis, :] - self.model.means_
            all_loglike[ii, :] = np.array([np.matmul(np.matmul(x, self.cov_inv), x.T) for x in v])

        all_loglike = np.min(all_loglike, axis=1)
        return np.sum(all_loglike) / data.shape[0]


class PearsonAsymptoticThresholdStrategy(ThresholdStrategy):

    def __init__(self, nbin):
        super().__init__()
        self.nbin = nbin

    def _get_threshold(self, alpha):
        return scipy.stats.chi2.isf(alpha, self.nbin - 1)
