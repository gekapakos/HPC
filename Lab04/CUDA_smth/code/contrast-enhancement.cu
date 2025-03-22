#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
/*#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            cleanup();
        } \
    } while (0)*/

/*void cleanup (void * gpuResult, )  {
    cudaFree(gpuResult);
}*/

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)  {
    PGM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    // printf("CPU\n");
    // for (int i = 0; i < 256; i++)  {
    //     printf(" %d",hist[i]);
    // }
    return result;
}

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in)  {
    cudaEvent_t startCuda, stopCuda;
    PGM_IMG gpuResult;
    PGM_IMG result;
    float millisecondsTransfers = 0, time;
    //int hist[256];
    int t_hist[256];
    int *d_hist;
    unsigned char * d_ImgIn;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    gpuResult.w = img_in.w;
    gpuResult.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda);


    cudaError_t err = cudaMalloc((void **)&gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
        //cleanup(gpuResult.img, result.img);
    }
    err = cudaMalloc((void **)&d_ImgIn, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
        //cleanup(gpuResult.img, result.img);
    }
    err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
        //cleanup(gpuResult.img, result.img);
    }
    
    //histogramGPU<<<1, 256>>>(d_hist);
    cudaMemset(d_hist, 0, sizeof(int) * 256);

    err = cudaMemcpy(d_ImgIn, img_in.img, gpuResult.w * gpuResult.h * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device
    

    histogramGPU<<<((gpuResult.h*gpuResult.w)/256)+1, 256, 256*sizeof(int)>>>(d_hist, d_ImgIn, gpuResult.w, gpuResult.h);

    err = cudaMemcpy(t_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device

    /*for (int i = 0; i < 256; i++)  {
        printf("Kapak: %d\n", t_hist[i]);
    }*/

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    printf("\nGPU1 Execution time: %lf seconds\n", millisecondsTransfers/1000.0);
    time = histogram_equalization_prep(gpuResult.img, img_in.img, t_hist, gpuResult.w, gpuResult.h, 256, d_ImgIn);

    time += millisecondsTransfers;

    cudaEventRecord(startCuda, 0);

    err = cudaMemcpy(result.img, gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);  // Copy data from host to device

    cudaFree(d_ImgIn);  
    cudaFree(d_hist);
    cudaFree(gpuResult.img);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    time += millisecondsTransfers;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return result;
}