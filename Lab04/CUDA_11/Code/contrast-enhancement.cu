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

    /*for(int i = 0;i < 256; i++) {
        printf("%d ", hist[i]);
    }*/

    return result;
}

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in)  {
    cudaEvent_t startCuda, stopCuda, startCuda1, stopCuda1;
    PGM_IMG gpuResult;
    PGM_IMG result;
    float millisecondsTransfers = 0, time, millisecondsTransfers1 = 0;
    //int hist[256];
    int t_hist[256], t_hist0[128], t_hist1[128];
    int *d_hist, *A0_hist, *A1_hist, *A2_hist, *A3_hist, *A4_hist, *A5_hist, *A6_hist, *A7_hist;
    // unsigned char * d_ImgIn;
    unsigned char * img_A0,* img_A1,* img_A2,* img_A3,* img_A4,* img_A5,* img_A6,* img_A7, *img_A[2];
    int SegSize = (img_in.w * img_in.h)/2;
    int t0_hist[256], t1_hist[256];
    cudaStream_t stream[2];
    

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    cudaMallocManaged(&gpuResult.img, result.w * result.h * sizeof(unsigned char));
    gpuResult.w = img_in.w;
    gpuResult.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);
    cudaEventCreate(&startCuda1);
    cudaEventCreate(&stopCuda1);

    cudaEventRecord(startCuda);


    /*cudaError_t err = cudaMalloc((void **)&gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }*/
    // cudaError_t err = cudaMalloc((void **)&d_ImgIn, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    cudaError_t err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    err = cudaMalloc((void**)&img_A[0], SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }

    err = cudaMalloc((void**)&img_A[1], SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    
    cudaMemset(d_hist, 0, sizeof(int) * 256);

    for (int i=0; i < 2; i++)  {
        int offset = i * SegSize;
        cudaStreamCreate(&stream[i]);
        //cudaMemcpyAsync(img_A[i], img_in.img+offset, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);
        histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream[i] >>>(d_hist, img_in.img+offset, SegSize);

        cudaMemcpyAsync(t_hist, d_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KapCUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    int *lut = histogram_equalization_prep(t_hist, 256, gpuResult.w * gpuResult.h);

    cudaEventRecord(startCuda1, 0);

    printf("\nGPU1 Execution time: %lf seconds\n", millisecondsTransfers/1000.0);
    int *lut_d;
    err = cudaMalloc((void**)&lut_d, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }

    for (int i=0; i < 2; i++)  {
        int offset = i * SegSize;
        cudaMemcpyAsync(lut_d, lut, 256*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        cudaBindTexture(0, texRef, lut_d, 256 * sizeof(int));
        histogram_equalization_GPU<<<((SegSize)/256), 256, 256*sizeof(int), stream[i]>>>(gpuResult.img + offset, img_in.img+offset, lut_d, img_in.w, img_in.h/2);

        //cudaMemcpyAsync(result.img + offset, gpuResult.img + offset, ((gpuResult.w * gpuResult.h)/2) * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[i]);
    }

    /*for(int i = 0;i < 256; i++) {
        printf("%d ", t_hist[i]);
    }*/

    //cudaFree(d_ImgIn);  
    cudaFree(d_hist);
    //cudaFree(gpuResult.img);

    cudaEventRecord(stopCuda1, 0);
    cudaUnbindTexture(texRef); // Unbind texture memory
    cudaEventSynchronize(stopCuda1);
    cudaEventElapsedTime(&millisecondsTransfers1, startCuda1, stopCuda1);

    millisecondsTransfers += millisecondsTransfers1;

    printf("\nGPU Execution time: %lf seconds\n", millisecondsTransfers/1000.0);

    return gpuResult;
}