#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int image_size) {
    int tx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + tx;
    extern __shared__ int private_hist[];

    if(tx < 256) {
        private_hist[tx] = 0;
    }

    // Constructs the Histogram Vector
    if (index < image_size)  
    {
        atomicAdd(&private_hist[img_in[index]], 1);
    }
    __syncthreads();

    if(tx < 256) {
        atomicAdd(&hist_out[tx], private_hist[tx]);
    }
}

texture<int, cudaTextureType1D, cudaReadModeElementType> lut_texture; // Bind the 1D texture

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    /* Get the result image */
    if (index < imageW * imageH) {
        // img_out[index] = tex1Dfetch(lut_texture, img_in[index]);
        img_out[index] = min(255, tex1Dfetch(lut_texture, img_in[index]));
    }
}

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
    PGM_IMG img_out;
    PGM_IMG result;
    float time = 0, time_new = 0;
    int hist[256];
    int *d_hist;
    unsigned char *img_A[2];
    int seg_size = (img_in.w * img_in.h)/2;
    int i, cdf, min, d, index;
    cudaStream_t stream[2];
    

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    cudaMallocManaged(&img_out.img, result.w * result.h * sizeof(unsigned char));

    img_out.w = img_in.w;
    img_out.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);
    cudaEventCreate(&startCudaNew);
    cudaEventCreate(&stopCudaNew);

    cudaEventRecord(startCuda);

    cudaError_t err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }
    err = cudaMalloc((void**)&img_A[0], seg_size * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }

    err = cudaMalloc((void**)&img_A[1], seg_size * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }
    
    cudaMemset(d_hist, 0, sizeof(int) * 256);

    for (int i=0; i < 2; i++)  {
        int offset = i * seg_size;
        cudaStreamCreate(&stream[i]);

        histogramGPU<<<((seg_size)/256)+1, 256, 256*sizeof(int), stream[i] >>>(d_hist, img_in.img+offset, seg_size);

        cudaMemcpyAsync(hist, d_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&time, startCuda, stopCuda);

    // int *lut = calculate_lut(hist, 256, img_out.w * img_out.h);
    int *lut = (int *)malloc(sizeof(int)*256);

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    // Finds the first value on the Histogram that isn't 0
    while(min == 0) {
        min = hist[i++];
    }
    index = i-1;

    // Calculate the look up table (lut)
    for (i = 0; i < index + 1; i++)  {
        lut[i] = 0;
    }

    d = result.w * result.h - min;
    for(i = index; i < 256; i++) {
        cdf += hist[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }
    for(i = 0; i < 256; i++) {
        if(lut[i] > 255) {
            lut[i] = 255;
        }
    }

    cudaEventRecord(startCudaNew, 0);

    printf("\nGPU1 Execution time: %lf seconds\n", time/1000.0);
    int *lut_d;
    err = cudaMalloc((void**)&lut_d, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }

    for (int i=0; i < 2; i++)  {
        int offset = i * seg_size;
        cudaMemcpyAsync(lut_d, lut, 256*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        cudaBindTexture(0, lut_texture, lut_d, 256 * sizeof(int));
        histogram_equalization_GPU<<<((seg_size)/256)+1, 256, 256*sizeof(int), stream[i]>>>(img_out.img + offset, img_in.img+offset, lut_d, img_in.w, img_in.h/2);
    }

    cudaFree(d_hist);

    cudaEventRecord(stopCudaNew, 0);
    cudaUnbindTexture(lut_texture); // Unbind texture memory
    cudaEventSynchronize(stopCudaNew);
    cudaEventElapsedTime(&time_new, startCudaNew, stopCudaNew);

    time += time_new;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return img_out;
}