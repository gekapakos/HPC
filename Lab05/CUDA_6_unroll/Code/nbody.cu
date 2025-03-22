#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
#define MAX_THREADS_PER_BLOCK 1024

typedef struct { float4 *position; float4 *velocity; } Body;

void randomizeBodies(float4 *position, float4 *velocity, int n) {
    for (int i = 0; i < n; i++) {
        position[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        position[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        position[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        velocity[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        velocity[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        velocity[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body p, float dt, int n) {
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    if (index >= n) return; // Guard against out-of-bounds

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float px = p.position[index].x;
    float py = p.position[index].y;
    float pz = p.position[index].z;

    __shared__ float4 position_s[MAX_THREADS_PER_BLOCK];

    for (int tile = 0; tile < gridDim.x; tile++) {
        __syncthreads();
        int tiledIndex = tile * blockDim.x + tx;
        position_s[tx] = p.position[tiledIndex];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < blockDim.x; j++) {
            float dx = position_s[j].x - px;
            float dy = position_s[j].y - py;
            float dz = position_s[j].z - pz;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads(); // Synchronize before next tile
    }

    p.velocity[index].x += dt * Fx;
    p.velocity[index].y += dt * Fy;
    p.velocity[index].z += dt * Fz;
}

int main(const int argc, const char **argv) {
    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);
    cudaEvent_t start_cuda, stop_cuda;

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(float4);
    float4 *position_h = (float4 *)malloc(bytes);
    float4 *velocity_h = (float4 *)malloc(bytes);
    Body p = { position_h, velocity_h };

    randomizeBodies(position_h, velocity_h, nBodies);

    float4 *position_d, *velocity_d;
    cudaMalloc(&position_d, bytes);
    cudaMalloc(&velocity_d, bytes);
    Body p_d = { position_d, velocity_d };

    int blocksPerGrid = (nBodies + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    double totalTime = 0.0;
    float time = 0.0;

    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    for (int iter = 1; iter <= nIters; iter++) {
        cudaEventRecord(start_cuda);
        cudaMemcpy(position_d, position_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(velocity_d, velocity_h, bytes, cudaMemcpyHostToDevice);
        bodyForce<<<blocksPerGrid, MAX_THREADS_PER_BLOCK>>>(p_d, dt, nBodies);
        cudaMemcpy(position_h, position_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(velocity_h, velocity_d, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);
        cudaEventElapsedTime(&time, start_cuda, stop_cuda);

        StartTimer();
        for (int i = 0; i < nBodies; i++) {
            p.position[i].x += p.velocity[i].x * dt;
            p.position[i].y += p.velocity[i].y * dt;
            p.position[i].z += p.velocity[i].z * dt;
        }

        const double tElapsed = GetTimer() / 1000.0 + time / 1000.0;
        if (iter > 1) totalTime += tElapsed;

        if (iter == 1) {
            FILE *outputFile = fopen("GPU.txt", "w");
            if (outputFile != NULL) {
              for (int i = 0; i < nBodies; i++) {
                fprintf(outputFile, "Body %d: x=%.6f, y=%.6f, z=%.6f\n",
                        i, p.position[i].x, p.position[i].y, p.position[i].z);
              }
              fclose(outputFile);
            } else {
              fprintf(stderr, "Error: Could not open output file.\n");
            }
        }

        printf("Iteration %d: %.5f seconds\n", iter, tElapsed);
    }

    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / (totalTime / (nIters - 1)));

    free(position_h);
    free(velocity_h);
    cudaFree(position_d);
    cudaFree(velocity_d);
}