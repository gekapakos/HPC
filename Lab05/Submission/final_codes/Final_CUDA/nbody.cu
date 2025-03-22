#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
#define MAX_THREADS_PER_BLOCK 1024

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} Body;

void randomizeBodies(float *x, float *y, float *z, float *vx, float *vy, float *vz, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body p, float dt, int n) {
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float px = p.x[index];
    float py = p.y[index];
    float pz = p.z[index];

    __shared__ float position_s[MAX_THREADS_PER_BLOCK][3];

    for (int tile = 0; tile < gridDim.x; tile++) {
        __syncthreads();
        int tiledIndex = tile * blockDim.x + tx;
        position_s[tx][0] = p.x[tiledIndex];
        position_s[tx][1] = p.y[tiledIndex];
        position_s[tx][2] = p.z[tiledIndex];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < blockDim.x; j++) {
            float dx = position_s[j][0] - px;
            float dy = position_s[j][1] - py;
            float dz = position_s[j][2] - pz;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads(); // Synchronize before next tile
    }

    p.vx[index] += dt * Fx;
    p.vy[index] += dt * Fy;
    p.vz[index] += dt * Fz;
}

int main(const int argc, const char **argv) {
    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);
    cudaEvent_t start_cuda, stop_cuda;

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(float);
    float *x_h = (float *)malloc(bytes);
    float *y_h = (float *)malloc(bytes);
    float *z_h = (float *)malloc(bytes);
    float *vx_h = (float *)malloc(bytes);
    float *vy_h = (float *)malloc(bytes);
    float *vz_h = (float *)malloc(bytes);
    Body p = { x_h, y_h, z_h, vx_h, vy_h, vz_h };

    randomizeBodies(x_h, y_h, z_h, vx_h, vy_h, vz_h, nBodies);

    float *x_d, *y_d, *z_d, *vx_d, *vy_d, *vz_d;
    cudaMalloc(&x_d, bytes);
    cudaMalloc(&y_d, bytes);
    cudaMalloc(&z_d, bytes);
    cudaMalloc(&vx_d, bytes);
    cudaMalloc(&vy_d, bytes);
    cudaMalloc(&vz_d, bytes);
    Body p_d = { x_d, y_d, z_d, vx_d, vy_d, vz_d };
    double totalTime = 0.0;
    float time = 0.0;

    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    int blocksPerGrid = (nBodies + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    for (int iter = 1; iter <= nIters; iter++) {
        cudaEventRecord(start_cuda);
        cudaMemcpy(x_d, x_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(z_d, z_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vx_d, vx_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vy_d, vy_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(vz_d, vz_h, bytes, cudaMemcpyHostToDevice);

        bodyForce<<<blocksPerGrid, MAX_THREADS_PER_BLOCK>>>(p_d, dt, nBodies);

        cudaMemcpy(x_h, x_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(y_h, y_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(z_h, z_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vx_h, vx_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy_h, vy_d, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vz_h, vz_d, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);
        cudaEventElapsedTime(&time, start_cuda, stop_cuda);

        StartTimer();
        for (int i = 0; i < nBodies; i++) {
            p.x[i] += p.vx[i] * dt;
            p.y[i] += p.vy[i] * dt;
            p.z[i] += p.vz[i] * dt;
        }

        const double tElapsed = GetTimer() / 1000.0 + time / 1000.0;
        if (iter > 1) totalTime += tElapsed;

        if (iter == 1) {
            FILE *outputFile = fopen("GPU.txt", "w");
            if (outputFile != NULL) {
              for (int i = 0; i < nBodies; i++) {
                fprintf(outputFile, "Body %d: x=%.6f, y=%.6f, z=%.6f\n",
                        i, p.x[i], p.y[i], p.z[i]);
              }
              fclose(outputFile);
            } else {
              fprintf(stderr, "Error: Could not open output file.\n");
            }
        }

        printf("Iteration %d: %.5f seconds\n", iter, tElapsed);
    }

    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / (totalTime / (nIters - 1)));

    free(x_h);
    free(y_h);
    free(z_h);
    free(vx_h);
    free(vy_h);
    free(vz_h);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaFree(vx_d);
    cudaFree(vy_d);
    cudaFree(vz_d);
}
