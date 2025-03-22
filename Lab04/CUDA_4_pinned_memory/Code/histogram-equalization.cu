#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
    int i;

    // Initialization in cpu
    for (i = 0; i < nbr_bin; i++) {
        hist_out[i] = 0;
    }

    // Constructs the Histogram Vector in gpu
    for (i = 0; i < img_size; i++) {
        //printf("CPUindex: %d, img: %d\n", i, img_in[i]);
        hist_out[img_in[i]]++;
    }
    
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index;

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    // Finds the first value on the Histogram that isn't 0
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i-1;

    // Calculate the look up table (lut)
    for (i = 0; i < index + 1; i++)  {
        lut[i] = 0;
    }

    d = img_size - min;
    for(i = index; i < nbr_bin; i++) {
        cdf += hist_in[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }

    for(i = 0; i < nbr_bin; i++)  {
       if(lut[i] > 255) {
            lut[i] = 255;
        }
    }
    
    /* Get the result image this is the only part of the function to be run in GPU */
    for(i = 0; i < img_size; i++) {
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
    free(lut);
}

__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int imageW, int imageH) {
    int tx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + tx;
    extern __shared__ int hist_shared[];

    if (tx < 256) {
        hist_shared[tx] = 0;
    }

    __syncthreads();

    // Constructs the Histogram Vector
    for(int i = index; i < (imageH * imageW); i += blockDim.x * gridDim.x) {
        atomicAdd(&hist_shared[img_in[i]], 1);
    }

    __syncthreads();

    if (tx< 256) {
        atomicAdd(&hist_out[tx], hist_shared[tx]);
    }
}

// texture<int, cudaTextureType1D, cudaReadModeElementType> texRef; // Bind the 1D texture

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int tx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + tx;
    extern __shared__ int lut_shared[];

    if(tx < 256) {
        lut_shared[tx] = lut[tx];
    }

    __syncthreads();

    /* Get the result image */
    if (index < (imageW * imageH)) {
        img_out[index] = (unsigned char) lut_shared[img_in[index]];
    }
}

int histogram_equalization_prep(unsigned char * img_out, unsigned char * img_in, int * hist_in, int imageW, int imageH, int nbr_bin, unsigned char * d_ImgIn) {
    int i, cdf, min, d, index, *d_lut;
    int img_size = imageW * imageH;
    float millisecondsTransfers = 0;
    cudaEvent_t startCuda, stopCuda;
    int *lut;
    cudaError_t err;
    // int *lut = (int *)malloc(sizeof(int)*nbr_bin);

    err = cudaHostAlloc((void**)&lut, nbr_bin*sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Host Alloc error: %s\n", cudaGetErrorString(err));
        if (lut) cudaFreeHost(lut);
        return(-1);
    }

    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    // Finds the first value on the Histogram that isn't 0
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i-1;

    // Calculate the look up table (lut)
    for (i = 0; i < index + 1; i++)  {
        lut[i] = 0;
    }

    d = img_size - min;
    for(i = index; i < nbr_bin; i++) {
        cdf += hist_in[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }
    for(i = 0; i < nbr_bin; i++)  {
        if(lut[i] > 255) {
            // printf("BLACK: %d\n", i);
            lut[i] = 255;
        }
    }

    // cudaMalloc((void **)&d_ImgIn, img_size * sizeof(unsigned char));

    // cudaMemcpy(d_ImgIn, img_in, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda, 0);

    cudaMalloc((void **)&d_lut, sizeof(int)*nbr_bin);

    cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);  // Copy data from host to device

    // cudaBindTexture(0, texRef, d_lut, 256 * sizeof(int));

    histogram_equalization_GPU<<<(img_size/256)+1, 256, 256 * sizeof(int)>>>(img_out, d_ImgIn, d_lut, imageW, imageH);
    cudaDeviceSynchronize(); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_lut);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    // cudaUnbindTexture(texRef); // Unbind texture memory
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    cudaFreeHost(lut);
    return(millisecondsTransfers);
}