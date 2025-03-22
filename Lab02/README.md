# Optimization Steps for Lab Assignment

This document summarizes optimization steps for a k-means clustering algorithm, starting with VTune profiling identifying `euclidean_distance` as a bottleneck. Optimizations begin with the `seq_kmeans` function and extend to smaller functions.

## 5.1 1st Optimization
- **Description**: Parallelizes the first `for` loop in `seq_kmeans` with OpenMP `parallel for`, using `private index`, `reduction delta`, `schedule(auto)`, and a `critical` section for `numClustersSize` and `newClusters`.
- **Outcome**: Basic parallelization, limited by the critical section.

## 5.2 2nd Optimization
- **Description**: Replaces `critical` with two `atomic` operations for additions on `numClustersSize` and `newClusters`. Inner loop variable `j` is set as `private`.
- **Outcome**: Faster synchronization improves execution time.

## 5.3 3rd Optimization
- **Description**: Removes `critical` and `atomic`, using `reduction` on `newClusterSize[numClusters]` to reduce thread blocking.
- **Outcome**: Enhanced performance by minimizing synchronization.

## 5.4 4th Optimization
- **Description**: Converts 2D `newClusters[numClusters][numCoords]` to 1D `newClusters[numClusters * numCoords]` for `reduction`, adjusting memory and access patterns.
- **Outcome**: Better scalability and performance.

## 5.5 5th Optimization
- **Description**: Applies `pragma simd` to the `for` loop in `euclid_dist_2` for CPU vectorization, as parallelization isn’t feasible.
- **Outcome**: Improved sequential performance.

## 5.6 6th Optimization
- **Description**: Tests `OMP_WAIT_POLICY` (ACTIVE vs. PASSIVE) and `OMP_PROC_BIND` for thread binding to optimize CPU usage and data locality.
- **Outcome**: Fine-tunes performance based on system configuration.
