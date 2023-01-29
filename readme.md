# Real-Time Clustering of Text Data for News Aggregation
This github repository contains the code that was used for the implementations of parallel solutions of the K-Means algorithm for the subject Parallel and Distributed Computing at FINKI 

- [Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

## Contents
### K-Means
- BaseKMeans.py - base class for all solutions
- SerialKMeans.py - serial implementation 
- MPICentroidKMeans.py - centroid-based implementation
- MPIBatchKMeans.py - batch-based implementation
- MPICentroidBatchKMeans.py - fusion-based implementation
### Additional
- helpers.py - used for encoding the text into vector
- autorun_centroid.sh - shell script for running the centroid experiments
- autorun_data.sh - shell script for running the data split experiments
- results/ - contains results from both experiments and .ipynb notebook for visualisation

## Prerequisites
- python 3.10
  - mpi4py
  - numpy
  - pandas
  - matplotlib