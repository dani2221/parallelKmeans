import time
import sys
import numpy as np
import json

from BaseKMeans import BaseKMeans
from mpi4py import MPI
from numpy.random import uniform


class MPICentroidKMeans(BaseKMeans):
    def __init__(self, k, max_iter, full_data):
        super().__init__(k, max_iter, full_data)

    def fit(self, data):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        centroids = np.zeros(self.k)
        if rank == 0:
            times = []
            full_time = time.time()
            min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
            centroids = np.array([uniform(min_, max_) for _ in range(self.k)])
        cluster_pos = np.zeros((1, len(data[0])))
        comm.Scatter(centroids, cluster_pos, root=0)
        cluster_pos = cluster_pos[0]
        iteration = 0

        if rank == 0:
            epoch_time = time.time()
        while iteration < self.max_iter:
            cluster = []
            distances = list()
            for x in data:
                distances.append(self.euclidean_vec(x, [cluster_pos])[0])

            distances = np.array(distances)
            comm.Barrier()
            min_distances = np.zeros_like(distances)

            comm.Allreduce(np.array(distances), min_distances, op=MPI.MIN)
            for i, (my_dist, min_dist) in enumerate(zip(distances, min_distances)):
                if my_dist == min_dist:
                    cluster.append(data[i])

            new_cluster_pos = np.mean(cluster, axis=0)
            if np.isnan(new_cluster_pos).any():
                new_cluster_pos = cluster_pos

            iteration += 1
            cluster_pos = new_cluster_pos

            if rank == 0:
                print(f"Iter: {str(iteration)}, Time: {time.time() - epoch_time}")
                times.append({'iter': iteration, 'time': time.time() - epoch_time})
                epoch_time = time.time()

        if rank == 0:
            print(f"Full time: {time.time() - full_time}")
            times.append({'iter': 'final', 'time': time.time() - full_time})
            with open('outputs-centroids/centroid_'+sys.argv[1] + '_' + sys.argv[2] + '.json', 'w') as fout:
                json.dump(times, fout)
        MPI.Finalize()


# usage: mpiexec python3 MPICentroidKMeans <num of centroids> (must be equal to num of processes)
if __name__ == "__main__":
    parallel = MPICentroidKMeans(k=int(sys.argv[1]), max_iter=10, full_data=float(sys.argv[2]))
    train = parallel.load_encodings()
    parallel.fit(train)
