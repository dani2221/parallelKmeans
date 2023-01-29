import numpy as np


class BaseKMeans:
    def __init__(self, k, max_iter, full_data=1):
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.full_data = full_data

    @staticmethod
    def euclidean_vec(point, data):
        return np.sqrt(np.sum((point - data) ** 2, axis=1))

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = self.euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs

    def load_encodings(self):
        with open('encodings.npy', 'rb') as f:
            encs = np.load(f)
        np.random.shuffle(encs)
        encs = encs[:int(len(encs)*self.full_data)]
        return encs