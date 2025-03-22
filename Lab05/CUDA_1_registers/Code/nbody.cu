#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define MAX_THREADS_PER_BLOCK 1024
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
  float px_reg = p[index].x, py_reg = p[index].y, pz_reg = p[index].z;

  for (int j = 0; j < n; j++) {
    float dx = p[j].x - px_reg;
    float dy = p[j].y - py_reg;
    float dz = p[j].z - pz_reg;
    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invDist = 1.0f / sqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
  }

  p[index].vx += dt*Fx; p[index].vy += dt*Fy; p[index].vz += dt*Fz;
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  cudaEvent_t start_cuda, stop_cuda;
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf_h = (float*)malloc(bytes);
  Body *p = (Body*)buf_h;

  randomizeBodies(buf_h, 6*nBodies); // Init pos / vel data

  float *buf_d;
  cudaMalloc(&buf_d, bytes);
  Body *p_d = (Body*)buf_d;

  int blocksPerGrid = (nBodies + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
  double totalTime = 0.0;
  float time = 0.0;

  cudaEventCreate(&start_cuda);
  cudaEventCreate(&stop_cuda);

  for (int iter = 1; iter <= nIters; iter++) {
    cudaEventRecord(start_cuda);
    cudaMemcpy(buf_d, buf_h, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<blocksPerGrid, MAX_THREADS_PER_BLOCK>>>(p_d, dt, nBodies); // compute interbody forces
    cudaMemcpy(buf_h, buf_d, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&time, start_cuda, stop_cuda);

    StartTimer();
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0 + time / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    
    if (iter == 1) {
      FILE *outputFile = fopen("GPU.txt", "w");
      if (outputFile != NULL) {
        for (int i = 0; i < nBodies; i++) {
          fprintf(outputFile, "Body %d: x=%.6f, y=%.6f, z=%.6f\n",
                  i, p[i].x, p[i].y, p[i].z);
        }
        fclose(outputFile);
      } else {
        fprintf(stderr, "Error: Could not open output file.\n");
      }
    }

    printf("Iteration %d: %e seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  // Free //
  free(buf_h);
  cudaFree(buf_d);
}