# Lab Exercises Repository

This repository contains the code and documentation for five laboratory exercises completed as part of a computational systems and parallel programming course. Each exercise focuses on different aspects of performance optimization, parallelization, and GPU programming. Below is a brief overview of each assignment.

## Exercise 1: Sequential Code Optimization

- **Description**: This exercise involves optimizing a sequential edge detection application using the Sobel filter on grayscale images. Starting with a provided baseline code, the task is to apply common code optimization techniques (e.g., loop interchange, loop unrolling, function inlining) step-by-step, measure execution time improvements, and validate correctness using PSNR. Experiments are conducted with and without compiler optimizations (Intel `icx` compiler, flags `-O0` and `-fast`).
- **Key Objectives**:
  - Stepwise optimization of a CPU-based sequential application.
  - Familiarity with experimentation and performance measurement.
  - Understanding compiler optimizations.

## Exercise 2: OpenMP Parallelization

- **Description**: This exercise focuses on parallelizing a sequential k-means clustering algorithm using OpenMP. The provided code is analyzed via profiling to identify parallelizable loops, which are then optimized with OpenMP directives. Performance is evaluated on a multicore Intel Xeon E5-2695 system with varying thread counts (1 to 56), using the Intel C compiler with optimization flags. The goal is to improve execution time while ensuring correctness compared to the sequential baseline.
- **Key Objectives**:
  - Gradual parallelization of a complete application.
  - Experimental evaluation on a multicore processor.

## Exercise 3: CUDA Programming Basics

- **Description**: This exercise introduces CUDA programming by implementing a 2D separable convolution filter (e.g., Sobel) on a GPU. Starting with a CPU-based reference code, the task involves porting the convolution to CUDA, experimenting with block/grid geometries, and addressing precision and thread count limitations. Performance and accuracy are compared between CPU and GPU implementations across various image sizes and filter radii.
- **Key Objectives**:
  - Familiarity with CUDA compilation (`nvcc`) and programming.
  - Experimentation with thread configurations and precision issues.

## Exercise 4: GPU-Based Histogram Equalization

- **Description**: This exercise involves implementing histogram equalization for contrast enhancement of grayscale images on a GPU. Starting from a sequential C implementation, the code is ported to CUDA, optimized using parallel patterns (e.g., parallel reductions), and tested with various image sizes. The focus is on achieving a fast and correct GPU implementation, with execution time as a competitive metric.
- **Key Objectives**:
  - Parallelization and optimization of an image processing application on GPU.
  - Ensuring functionality across arbitrary image sizes.

## Exercise 5: N-Body Simulation on CPU and GPU

- **Description**: This exercise implements an N-Body simulation modeling physical interactions (e.g., gravitational forces) among particles. The provided sequential C code is first parallelized for CPU using OpenMP, then ported to GPU with CUDA. Optimization techniques (e.g., tiling, unrolling) are applied to enhance GPU performance, with correctness verified against CPU results. Performance is measured with 128K bodies.
- **Key Objectives**:
  - Parallelization and optimization of a dynamic simulation.
  - Comparison of CPU (OpenMP) and GPU (CUDA) implementations.

## Repository Structure

- `Lab01/`: Sequential optimization code and report.
- `Lab02/`: OpenMP parallelized k-means code and evaluation.
- `Lab03/`: CUDA convolution code and experimental results.
- `Lab04/`: GPU histogram equalization code.
- `Lab05/`: N-Body simulation code for CPU (OpenMP) and GPU (CUDA).

## Compilation Instructions

- **Exercise 1**: Use `icx -Wall -O0/-fast <source.c> -o <executable>`.
- **Exercise 2**: Use Intel C compiler with OpenMP support (e.g., `icx -fast -qopenmp`).
- **Exercise 3 & 4**: Use `nvcc -arch=sm_37 -o <executable> <source.cu>`.
- **Exercise 5**: 
  - CPU: `gcc -std=c99 -O3 -fopenmp -o nbody nbody.c -lm`
  - GPU: `nvcc -arch=sm_37 -o nbody nbody.cu`

## Notes

- All exercises were developed and tested on specific lab systems (e.g., `inf-mars1`, `csl-venus`).
- Detailed reports and experimental results are included where required.
