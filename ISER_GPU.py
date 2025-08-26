import numpy as np
from random import sample
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.ensemble import IsolationForest
import warnings
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import torch

warnings.filterwarnings('ignore', category=RuntimeWarning)

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

class ISER:    
    def __init__(self, max_samples=16, n_estimators=200, if_max_samples=256, random_state=None, novelty=True, use_gpu=False):
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None
        self.if_max_samples = if_max_samples
        self.novelty = novelty
        self.use_gpu = use_gpu

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
            
            self._dratio[i] = 1 - 1/(self._centroids_radius_array[i] + MIN_FLOAT)

        if not self.novelty:
            self.data_transformed = self.transform(self.data)

        return self

    def transform(self, X):
        check_is_fitted(self, ['_centroids_radius_array', '_dratio'])
        
        if self.use_gpu and torch.cuda.is_available():
            return self._transform_gpu(X)
        else:
            return self._transform_cpu(X)
    
    def _transform_cpu(self, X):
        iso_map = np.ones([X.shape[0], self.n_estimators])

        for i in range(self.n_estimators):
            x_dists = euclidean_distances(X, self._centroids[i], squared=True)            
            cover_radius = np.where(
                x_dists <= self._centroids_radius_array[i],
                self._centroids_radius_array[i], np.nan)
            x_covered = np.where(~np.isnan(cover_radius).all(axis=1))
            if len(x_covered[0]) > 0:
                cnn_x = np.nanargmin(cover_radius[x_covered], axis=1)
                iso_map[x_covered, i] = self._dratio[i][cnn_x]

        return iso_map
    
    def _transform_gpu(self, X):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        centroids_torch = torch.tensor(self._centroids, dtype=torch.float32, device=device)
        radius_torch = torch.tensor(self._centroids_radius_array, dtype=torch.float32, device=device)
        dratio_torch = torch.tensor(self._dratio, dtype=torch.float32, device=device)
        
        iso_map = torch.ones([X.shape[0], self.n_estimators], device=device)

        for i in range(self.n_estimators):
            diffs = X_torch.unsqueeze(1) - centroids_torch[i].unsqueeze(0)
            x_dists = torch.sum(diffs**2, dim=2)
            
            cover_radius = torch.where(
                x_dists <= radius_torch[i],
                radius_torch[i], 
                torch.tensor(float('nan'), device=device)
            )
            
            x_covered = torch.where(~torch.isnan(cover_radius).all(dim=1))[0]
            
            if len(x_covered) > 0:
                cover_subset = cover_radius[x_covered]
                cover_subset_clean = cover_subset.clone()
                cover_subset_clean[torch.isnan(cover_subset_clean)] = 1e10
                cnn_x = torch.argmin(cover_subset_clean, dim=1)
                iso_map[x_covered, i] = dratio_torch[i][cnn_x]

        return iso_map.cpu().numpy()

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