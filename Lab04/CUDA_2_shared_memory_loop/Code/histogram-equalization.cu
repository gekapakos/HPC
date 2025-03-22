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

__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int imageW, int imageH) 
{
    int tx = threadIdx.x;
    int index = blockIdx.x*blockDim.x + tx;
    extern __shared__ int private_hist[];

    if(tx < 256) {
        private_hist[tx] = 0;
    }

    // Constructs the Histogram Vector
    // Constructs the Histogram Vector
    for(int i = index; i < (imageH * imageW); i += blockDim.x * gridDim.x) {
        atomicAdd(&private_hist[img_in[i]], 1);
    }
    __syncthreads();

    if(tx < 256) {
        atomicAdd(&hist_out[tx], private_hist[tx]);
    }
}

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) 
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    /* Get the result image */
    if (index < imageW * imageH)  
    {
        img_out[index] = (unsigned char)lut[img_in[index]];
    }
    __syncthreads();
}

int* lut_computation(int * hist_in, int nbr_bin, int img_size) {
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

    return lut;
}