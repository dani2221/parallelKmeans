#!/bin/zsh
echo "serial"
python3 SerialKMeans.py 5 0.01
python3 SerialKMeans.py 5 0.1
python3 SerialKMeans.py 5 0.25
python3 SerialKMeans.py 5 0.5
python3 SerialKMeans.py 5 0.75
python3 SerialKMeans.py 5 1

echo "centroid"
mpiexec -n 5 python3 MPICentroidKMeans.py 5 0.01
mpiexec -n 5 python3 MPICentroidKMeans.py 5 0.1
mpiexec -n 5 python3 MPICentroidKMeans.py 5 0.25
mpiexec -n 5 python3 MPICentroidKMeans.py 5 0.5
mpiexec -n 5 python3 MPICentroidKMeans.py 5 0.75
mpiexec -n 5 python3 MPICentroidKMeans.py 5 1

echo "batch"
mpiexec -n 16 python3 MPIBatchKMeans.py 5 0.01
mpiexec -n 16 python3 MPIBatchKMeans.py 5 0.1
mpiexec -n 16 python3 MPIBatchKMeans.py 5 0.25
mpiexec -n 16 python3 MPIBatchKMeans.py 5 0.5
mpiexec -n 16 python3 MPIBatchKMeans.py 5 0.75
mpiexec -n 16 python3 MPIBatchKMeans.py 5 1

echo "fusion"
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 0.01
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 0.1
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 0.25
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 0.5
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 0.75
mpiexec -n 10 python3 MPICentroidBatchKMeans.py 5 2 1

mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 0.01
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 0.1
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 0.25
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 0.5
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 0.75
mpiexec -n 15 python3 MPICentroidBatchKMeans.py 5 3 1
