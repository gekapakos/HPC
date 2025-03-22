# N-Body Simulation Optimizations

This outlines CPU and GPU optimizations for an N-Body simulation with O(N²) complexity.

## CPU Baseline
- **Description**: Sequential CPU code, poor performance due to O(N²) scaling.

## OpenMP
- **OMP 1st Approach**: Parallelizes outer loop with `#pragma omp parallel for`, using `private` (distSqr, invDist, j), `reduction` (Fx, Fy, Fz), and static scheduling.
  - **Outcome**: x25.96 speedup vs. baseline.
- **OMP 2nd Approach**: Removes unneeded `private`/`reduction` directives, keeping only `#pragma omp parallel for`.
  - **Outcome**: x0.944 vs. 1st approach, slightly worse; x1.0008 better at 56 threads (131072 bodies).

## GPU Implementation
- **CUDA 0 Baseline**: Ports `bodyForce` to GPU, one thread per body, with CPU-GPU transfers.
  - **Outcome**: x2.36 speedup vs. 2nd OpenMP.
- **CUDA 1 Registers**: Stores position in registers, not global memory.
  - **Outcome**: x0.99, slightly slower.
- **CUDA 2 Struct of Arrays (SoA)**: Uses SoA for coalesced memory access.
  - **Outcome**: x0.947 vs. registers, worse due to scattered access.
- **CUDA 3 Float3**: Converts to 3D vectors (x, y, z).
  - **Outcome**: x1.044 vs. SoA float, x0.989 vs. registers.
- **CUDA 4 Float4**: Uses 4D vectors for 16-byte alignment.
  - **Outcome**: x1.042 vs. float3, x1.03 vs. registers.
- **CUDA 5 Tiling**: Uses shared memory tiles to cut global accesses.
  - **Outcome**: x1.195 vs. float4 SoA.
- **CUDA 6 Loop Unrolling**: Unrolls inner loop for parallel execution.
  - **Outcome**: x1.002 vs. tiling.
- **CUDA 7 Pinned Memory**: Skipped; `nvprof` shows low memcpy overhead.
  - **Outcome**: Avoided due to compute-bound nature.
- **CUDA 8 Flags & Fast Math**: Adds `-ftz=true` (flushes denormals) and `rsqrtf()` (fast reciprocal sqrt).
  - **Outcome**: x2.236 vs. unrolled, major speedup
