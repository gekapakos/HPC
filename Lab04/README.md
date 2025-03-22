# Optimizations for Lab Assignment

This outlines CPU and GPU optimizations for histogram equalization on images.

## CPU
- **Description**: Row-wise traversal for histogram and equalization, O(N) complexity. Divergence removed in:
  - **3.1.1**: First loop avoids `if` by initializing `lut` to zero until first non-zero histogram value, using CDF.
  - **3.1.2**: Second loop thresholds `lut` to 255 in a 256-bin loop, then assigns output directly.
- **Outcome**: Cleaner code, better for parallelization.

## Default GPU
- **Description**: Naive CUDA port of histogram and equalization with `atomicAdd` and `syncthreads` for sync.
- **Outcome**: Faster than CPU due to coalesced memory access; scales with image size.

## GPU - CUDA 1 & CUDA 2
- **CUDA 1 Shared Memory**: Uses 256 threads/block for 256-bin histogram in fast shared memory, then copies to global.
  - **Outcome**: Boosts small images, less effective for large ones due to sync overhead.
- **CUDA 2 Shared Memory with Loop**: Strides with 256 threads, uses `atomicAdd` for shared/global updates.
  - **Outcome**: Worse for large images due to sync contention.

## Histogram Equalization GPU - CUDA 3
- **Description**: Shared memory for `lut`, copied from global, then used for output.
- **Outcome**: Improves small images, struggles with large ones.

## Pinned Memory - CUDA 4 & CUDA 5
- **CUDA 4 cudaHostAllocDefault**: Pinned host memory for fast transfers.
  - **Outcome**: Speeds up all image sizes via direct access.
- **CUDA 5 cudaHostAllocMapped**: Mapped to GPU address space for zero-copy.
  - **Outcome**: Slower than default due to PCIe latency.

## Unified Memory - CUDA 7
- **Description**: Uses `cudaMallocManaged` for shared CPU/GPU memory.
- **Outcome**: Simplifies code, outperforms pinned mapped memory.

## Constant Memory - CUDA 8
- **Description**: `lut` in 64KB cached constant memory for fast, read-only access.
- **Outcome**: Outperforms shared memory for uniform access.

## Texture Memory - CUDA 9
- **Description**: `lut` in cached texture memory for spatial locality.
- **Outcome**: Slightly worse than constant memory for 1D data.

## Streams CUDA 6
- **Description**: Two streams split image halves for concurrent execution.
- **Outcome**: May underperform due to resource contention.
