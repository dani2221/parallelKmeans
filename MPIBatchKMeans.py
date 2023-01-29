import time
import sys
from mpi4py import MPI
from numpy.random import uniform
import numpy as np
from BaseKMeans import BaseKMeans
import json


class MPIBatchKMeans(BaseKMeans):
    def __init__(self, k, max_iter, full_data):
        super().__init__(k, max_iter, full_data)
        self.centroids = None

    @staticmethod
    def split_batches(a, n):
        k, m = divmod(len(a), n)
        return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def fit(self, data):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            times = []
            full_time = time.time()
            min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
            self.centroids = [uniform(min_, max_) for _ in range(self.k)]
        self.centroids = comm.bcast(self.centroids, root=0)
        data_split = comm.scatter(self.split_batches(data, size), root=0)
        iteration = 0

        if rank == 0:
            epoch_time = time.time()

        while iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.k)]
            for x in data_split:
                dists = self.euclidean_vec(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids
            full_centroids = comm.gather(sorted_points, root=0)
            comm.Barrier()
            if rank == 0:
                merged_pts = [[] for el in range(self.k)]
                for split in full_centroids:
                    for i in range(len(split)):
                        merged_pts[i] += split[i]

                self.centroids = [np.mean(cluster, axis=0) for cluster in merged_pts]
                for i, centroid in enumerate(self.centroids):
                    if np.isnan(centroid).any():
                        self.centroids[i] = prev_centroids[i]
            self.centroids = comm.bcast(self.centroids, root=0)

            comm.Barrier()
            iteration += 1
            if rank == 0:
                print(f"Iter: {str(iteration)}, Time: {time.time()-epoch_time}")
                times.append({'iter': iteration, 'time': time.time() - epoch_time})
                epoch_time = time.time()

        if rank == 0:
            print(f"Full time: {time.time()-full_time}")
            times.append({'iter': 'final', 'time': time.time() - full_time})
            with open('outputs-centroids/batch_' + sys.argv[1] + '_' + sys.argv[2] + '.json', 'w') as fout:
                json.dump(times, fout)


# usage: mpiexec python3 MPIBatchKMeans <num of centroids>
if __name__ == "__main__":
    parallel = MPIBatchKMeans(k=int(sys.argv[1]), max_iter=10, full_data=float(sys.argv[2]))
    train = parallel.load_encodings()
    parallel.fit(train)