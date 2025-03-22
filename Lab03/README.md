# CUDA Optimizations for Lab Assignment

This outlines CUDA optimizations for a convolution image processing task.

## Modification 1
- **Description**: Allocate GPU memory, copy CPU inputs, run CPU/GPU (1 block: width × height threads), compare results, free memory.
- **Outcome**: Max 32×32 images due to 1024-thread limit.

## Modification 2
- **Description**: Grid: `imageW/32` × `imageH/32`, supports 16384×16384 (3.32 GB).
- **Outcome**: Scales better, uses GPU fully.

## Modification 3
- **Description**: Uses doubles for precision, ups memory/performance cost.
- **Outcome**: Better accuracy (10 at 1, 5 at 256), slower runtime.
- **Figures**: Fig. 11-13, Table 3.

## Modification 4
- **Description**: Adds `FILTER_RADIUS` zero-padding to edges.
- **Outcome**: Fixes edge issues, simplifies kernel.
