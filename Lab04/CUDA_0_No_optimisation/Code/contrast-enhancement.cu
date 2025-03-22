#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)  {
    PGM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    return result;
}

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in)  {
    cudaEvent_t startCuda, stopCuda, startCudaNew, stopCudaNew;
    PGM_IMG img_gpu;
    PGM_IMG img_cpu;
    float time = 0, timeNew;
    int hist[256];
    int *d_hist, *d_lut;
    unsigned char * d_ImgIn;

    img_cpu.w = img_in.w;
    img_cpu.h = img_in.h;
    img_cpu.img = (unsigned char *)malloc(img_cpu.w * img_cpu.h * sizeof(unsigned char));
    img_gpu.w = img_in.w;
    img_gpu.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda);


    cudaError_t err = cudaMalloc((void **)&img_gpu.img, img_gpu.w * img_gpu.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_gpu.img) cudaFree(img_gpu.img);
        return(img_gpu);
    }
    err = cudaMalloc((void **)&d_ImgIn, img_gpu.w * img_gpu.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_gpu.img) cudaFree(img_gpu.img);
        return(img_gpu);
    }
    err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_gpu.img) cudaFree(img_gpu.img);
        return(img_gpu);
    }
    
    cudaMemset(d_hist, 0, sizeof(int) * 256);

    err = cudaMemcpy(d_ImgIn, img_in.img, img_cpu.w * img_cpu.h * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device
    

    histogramGPU<<<((img_cpu.w*img_cpu.h)/1024), 1024 >>>(d_hist, d_ImgIn, img_cpu.w, img_cpu.h);

    err = cudaMemcpy(hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&time, startCuda, stopCuda);

    printf("\nGPU1 Execution time: %lf seconds\n", time/1000.0);
    cudaEventCreate(&startCudaNew);
    cudaEventCreate(&stopCudaNew);

    int *lut = lut_computation(hist, 256, img_cpu.w * img_cpu.h);

    cudaEventRecord(startCudaNew, 0);

    cudaMalloc((void **)&d_lut, sizeof(int)*256);

    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);  // Copy data from host to device

    histogram_equalization_GPU<<<((img_cpu.w * img_cpu.h)/1024), 1024>>>(img_gpu.img, d_ImgIn, d_lut, img_cpu.w, img_cpu.h);
    cudaDeviceSynchronize(); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(img_cpu.img, img_gpu.img, img_cpu.w * img_cpu.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);  // Copy data from host to device

    cudaFree(d_ImgIn);  
    cudaFree(d_hist);
    cudaFree(img_gpu.img);
    cudaFree(d_lut);

    cudaEventRecord(stopCudaNew, 0);
    cudaEventSynchronize(stopCudaNew);
    cudaEventElapsedTime(&timeNew, startCudaNew, stopCudaNew);

    time += timeNew;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return img_cpu;
}