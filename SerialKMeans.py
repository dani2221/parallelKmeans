import sys

import numpy as np
from numpy.random import uniform
from BaseKMeans import BaseKMeans
import time
import json


class SerialKMeans(BaseKMeans):
    def __init__(self, k, max_iter, full_data):
        super().__init__(k, max_iter, full_data)
        self.centroids = None

    def fit(self, data):
        times = []
        full_time = time.time()
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.k)]
        prev_centroids = None

        epoch_time = time.time()
        iteration = 0
        while iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.k)]
            for x in data:
                dists = self.euclidean_vec(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
            print(f"Iter: {str(iteration)}, Time: {time.time()-epoch_time}")
            times.append({'iter': iteration, 'time': time.time() - epoch_time})
            epoch_time = time.time()

        print(f"Full time: {time.time()-full_time}")
        times.append({'iter': 'final', 'time': time.time() - full_time})
        with open('outputs-centroids/serial_' + sys.argv[1] + '_' + sys.argv[2] + '.json', 'w') as fout:
            json.dump(times, fout)


if __name__ == "__main__":
    serial = SerialKMeans(k=int(sys.argv[1]), max_iter=10, full_data=float(sys.argv[2]))
    serial.fit(serial.load_encodings())
