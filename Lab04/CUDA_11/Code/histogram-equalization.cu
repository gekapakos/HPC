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

__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int image_size) {
    extern __shared__ int sharedMemory[];
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    for (unsigned int bin = tx; bin < 256; bin += blockDim.x) {
        sharedMemory[bin] = 0u;
    }
    __syncthreads();

    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    
    for(unsigned int i = index; i < image_size; i+=blockDim.x*gridDim.x) {
        int alphabet_posisition = img_in[i] - 'a';
        if(alphabet_posisition >= 0 && alphabet_posisition < 26)
        {
            int bin = alphabet_posisition/4;

            if(bin == prevBinIdx) {
                ++accumulator;
            }
            else {
                if(accumulator > 0) {
                    atomicAdd(&sharedMemory[prevBinIdx], accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if(accumulator > 0) {
        atomicAdd(&sharedMemory[prevBinIdx], accumulator);
    }
    __syncthreads();

    for(unsigned int bin = tx; bin < 256; bin += blockDim.x) {
        unsigned int binValue = sharedMemory[bin];
        if(binValue > 0) {
            atomicAdd(&sharedMemory[bin], binValue);
        }
    }
}

// texture<int, cudaTextureType1D, cudaReadModeElementType> texRef; // Bind the 1D texture

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    /* Get the result image */
    if ((index) < imageW * imageH)  {
        img_out[index] = tex1Dfetch(texRef, img_in[index]); // cudaBindTexture(0, texRef, lut_d, 256 * sizeof(int));
    }
    __syncthreads();
}

int* histogram_equalization_prep(int * hist_in, int nbr_bin, int img_size) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index, *d_lut;
    float millisecondsTransfers = 0;
    // cudaEvent_t startCuda, stopCuda;

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

    return lut;
}