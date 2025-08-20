import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_random_state
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
import warnings
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore', category=RuntimeWarning)


MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

class ISER:    
    def __init__(self, max_samples=16, n_estimators=200, if_max_samples = 256,random_state=None, novelty = True):
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None
        self.if_max_samples = if_max_samples
        self.novelty = novelty

    def fit(self, data):
        self.data = data
        n_samples, n_features = data.shape
        self.centroid = []

        self._centroids = np.empty([self.n_estimators, self.max_samples, n_features])
        self._centroids_radius_array = np.empty([self.n_estimators, self.max_samples])
        self._dratio = np.empty([self.n_estimators, self.max_samples])

        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            subIndex = rnd.choice(n_samples, self.max_samples, replace=False)
            self.centroid.append(subIndex)
            
            tdata = self.data[subIndex, :]
            self._centroids[i] = tdata 

            center_dist = euclidean_distances(tdata, tdata, squared=True)
            np.fill_diagonal(center_dist, np.inf)
            
            self._centroids_radius_array[i] = np.amin(center_dist, axis=1)
            
            self._dratio[i] = 1- 1/(self._centroids_radius_array[i] + MIN_FLOAT)

        self.data_transformed = self.transform(self.data)

        self.model_ = IsolationForest(
            max_samples=self.if_max_samples, 
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model_.fit(self.data_transformed)
        
        return self
    

    def transform(self, X):
        check_is_fitted(self, ['_centroids_radius_array', '_dratio'])
        
        iso_map = np.ones([X.shape[0], self.n_estimators])

        for i in range(self.n_estimators):
            x_dists = euclidean_distances(X, self._centroids[i], squared=True)            
            cover_radius = np.where(
                x_dists <= self._centroids_radius_array[i],
                self._centroids_radius_array[i], np.nan)
            x_covered = np.where(~np.isnan(cover_radius).all(axis=1))
            cnn_x = np.nanargmin(cover_radius[x_covered], axis=1)
            iso_map[x_covered, i] = self._dratio[i][cnn_x]

        return iso_map


    def ISER_A(self, X):
        if self.novelty:
            X_transformed = self.transform(X)
            scores = np.mean(X_transformed, axis=1)
        else:
            scores = np.mean(self.data_transformed, axis=1)
        return scores



    def ISER_S(self, X):
        one = np.ones(self.n_estimators)
        one = one.reshape(1, -1)
        if self.novelty:
            X_transformed = self.transform(X)
            scores = cosine_similarity(X_transformed, one)
        else:
            scores = cosine_similarity(self.data_transformed, one)
        return scores


    
    def ISER_IF(self, X):
        if self.novelty:
            X_transformed = self.transform(X)
            score = self.model_.decision_function(X_transformed)
        else:
            score = self.model_.decision_function(self.data_transformed)
        return score