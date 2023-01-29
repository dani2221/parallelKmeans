#!/bin/zsh
echo "serial"
python3 SerialKMeans.py 1 1
python3 SerialKMeans.py 3 1
python3 SerialKMeans.py 5 1
python3 SerialKMeans.py 7 1
python3 SerialKMeans.py 9 1
python3 SerialKMeans.py 11 1
python3 SerialKMeans.py 13 1
python3 SerialKMeans.py 15 1
python3 SerialKMeans.py 16 1

echo "centroid"
mpiexec -n 1 python3 MPICentroidKMeans.py 1 1
mpiexec -n 3 python3 MPICentroidKMeans.py 3 1
mpiexec -n 5 python3 MPICentroidKMeans.py 5 1
mpiexec -n 7 python3 MPICentroidKMeans.py 7 1
mpiexec -n 9 python3 MPICentroidKMeans.py 9 1
mpiexec -n 11 python3 MPICentroidKMeans.py 11 1
mpiexec -n 13 python3 MPICentroidKMeans.py 13 1
mpiexec -n 15 python3 MPICentroidKMeans.py 15 1
mpiexec -n 16 python3 MPICentroidKMeans.py 16 1

echo "batch"
mpiexec -n 16 python3 MPIBatchKMeans.py 1 1
mpiexec -n 16 python3 MPIBatchKMeans.py 3 1
mpiexec -n 16 python3 MPIBatchKMeans.py 5 1
mpiexec -n 16 python3 MPIBatchKMeans.py 7 1
mpiexec -n 16 python3 MPIBatchKMeans.py 9 1
mpiexec -n 16 python3 MPIBatchKMeans.py 11 1
mpiexec -n 16 python3 MPIBatchKMeans.py 13 1
mpiexec -n 16 python3 MPIBatchKMeans.py 15 1
mpiexec -n 16 python3 MPIBatchKMeans.py 16 1

echo "fusion"
mpiexec -n 2 python3 MPICentroidBatchKMeans.py 1 2 1
mpiexec -n 4 python3 MPICentroidBatchKMeans.py 2 2 1
mpiexec -n 8 python3 MPICentroidBatchKMeans.py 4 2 1
mpiexec -n 12 python3 MPICentroidBatchKMeans.py 6 2 1
mpiexec -n 16 python3 MPICentroidBatchKMeans.py 8 2 1

mpiexec -n 3 python3 MPICentroidBatchKMeans.py 1 3 1
mpiexec -n 6 python3 MPICentroidBatchKMeans.py 2 3 1
mpiexec -n 9 python3 MPICentroidBatchKMeans.py 3 3 1
mpiexec -n 12 python3 MPICentroidBatchKMeans.py 4 3 1
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 1

mpiexec -n 4 python3 MPICentroidBatchKMeans.py 1 4 1
mpiexec -n 8 python3 MPICentroidBatchKMeans.py 2 4 1
mpiexec -n 12 python3 MPICentroidBatchKMeans.py 3 4 1
mpiexec -n 16 python3 MPICentroidBatchKMeans.py 4 4 1
