import time
import sys
from mpi4py import MPI
from numpy.random import uniform
import numpy as np
from BaseKMeans import BaseKMeans
import json


class MPICentroidBatchKMeans(BaseKMeans):
    def __init__(self, k, max_iter, data_processes, full_data):
        super().__init__(k, max_iter, full_data)
        self.data_processes = data_processes

    @staticmethod
    def split_batches(a, n):
        k, m = divmod(len(a), n)
        return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def fit(self, data):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        centroids = np.zeros(self.k)
        cluster_pos = None
        if rank == 0:
            times = []
            full_time = time.time()
            min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
            centroids = np.array([uniform(min_, max_) for _ in range(self.k)])
            cluster_pos = centroids[0]
            for i in range(self.k):
                for j in range(self.data_processes):
                    if i == j and i == 0:
                        continue
                    comm.send(centroids[i], i * self.data_processes + j)
        else:
            cluster_pos = comm.recv(source=0)
        iteration = 0

        data_split = self.split_batches(data, self.data_processes)[rank % self.data_processes]

        if rank == 0:
            epoch_time = time.time()

        while iteration < self.max_iter:
            cluster = []
            dists = []
            for x in data_split:
                dists.append(self.euclidean_vec(x, [cluster_pos])[0])

            min_dists = np.array(dists)
            if rank in range(self.data_processes):
                for i in range(rank + self.data_processes, self.k * self.data_processes, self.data_processes):
                    min_dists = np.minimum(min_dists, comm.recv(source=i))
                for i in range(rank + self.data_processes, self.k * self.data_processes, self.data_processes):
                    comm.send(min_dists, i)
            else:
                comm.send(dists, rank % self.data_processes)
                min_dists = comm.recv(source=rank % self.data_processes)

            for i, (my_dist, min_dist) in enumerate(zip(np.array(dists), min_dists)):
                if my_dist == min_dist:
                    cluster.append(data[i])

            if rank % self.data_processes == 0:
                full_data = [] + cluster
                for i in range(rank + 1, rank + self.data_processes):
                    full_data += comm.recv(source=i)

                new_cluster_pos = np.mean(full_data, axis=0)
                if np.isnan(new_cluster_pos).any():
                    new_cluster_pos = cluster_pos
                for i in range(rank + 1, rank + self.data_processes):
                    comm.send(new_cluster_pos, i)
            else:
                comm.send(cluster, rank - rank % self.data_processes)
                new_cluster_pos = comm.recv(source=rank - rank % self.data_processes)

            cluster_pos = new_cluster_pos
            comm.Barrier()
            iteration += 1
            if rank == 0:
                print(f"Iter: {str(iteration)}, Time: {time.time() - epoch_time}")
                times.append({'iter': iteration, 'time': time.time() - epoch_time})
                epoch_time = time.time()

        if rank == 0:
            print(f"Full time: {time.time() - full_time}")
            times.append({'iter': 'final', 'time': time.time() - full_time})
            with open('outputs-centroids/centroidbatch_'+sys.argv[1]+'-'+sys.argv[2] + '_' + sys.argv[3] + '.json', 'w') as fout:
                json.dump(times, fout)
        MPI.Finalize()


# usage: mpiexec python3 MPICentroidKMeans <num of centroids> <data_split> (num*data must be num of processes)
if __name__ == "__main__":
    parallel = MPICentroidBatchKMeans(k=int(sys.argv[1]),
                                      max_iter=10,
                                      data_processes=int(sys.argv[2]),
                                      full_data=float(sys.argv[3]))
    train = parallel.load_encodings()
    parallel.fit(train)


