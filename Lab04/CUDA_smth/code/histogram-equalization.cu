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
    extern __shared__ int sharedMemory[];
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    if (tx < 256) {
        sharedMemory[tx] = 0;
    }

    __syncthreads();

    // Constructs the Histogram Vector
    if (index < imageH*imageW)  {
        atomicAdd(&sharedMemory[img_in[index]], 1);
        __syncthreads();
        atomicAdd(&hist_out[tx], sharedMemory[tx]);
    }
    __syncthreads();
}

texture<int, cudaTextureType1D, cudaReadModeElementType> texRef; // Bind the 1D texture

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int y = index / imageW; // row
    int x = index % imageW; // col
    
    /* Get the result image */

    if ((y * imageW + x) < imageW * imageH)  {
        img_out[index] = tex1Dfetch(texRef, img_in[index]);
    }
    //printf("ABLACK: %d\n", img_out[y*imageW+x]);

}

// __global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
//     int index = blockIdx.x*blockDim.x + threadIdx.x;
//     int y = index / imageW; // row
//     int x = index % imageW; // col
//     extern __shared__ unsigned char cuChulain[];
//     /* Get the result image */

//     if (threadIdx.x < 256)  {
//         cuChulain[threadIdx.x] =  lut[threadIdx.x];
//     }
    
//     __syncthreads();

//     if ((y * imageW + x) < imageW * imageH)  {
//         img_out[y*imageW+x] = cuChulain[img_in[y*imageW+x]];
//     }
//     __syncthreads();
//     //printf("ABLACK: %d\n", img_out[y*imageW+x]);

// }

int histogram_equalization_prep(unsigned char * img_out, unsigned char * img_in, int * hist_in, int imageW, int imageH, int nbr_bin, unsigned char * d_ImgIn) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index, *d_lut;
    int img_size = imageW * imageH;
    float millisecondsTransfers = 0;
    cudaEvent_t startCuda, stopCuda;

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

    cudaBindTexture(0, texRef, d_lut, 256 * sizeof(int));

    // histogram_equalization_GPU<<<(img_size/256)+1, 256, 256 * sizeof(unsigned char)>>>(img_out, d_ImgIn, d_lut, imageW, imageH);
    histogram_equalization_GPU<<<(img_size/256)+1, 256>>>(img_out, d_ImgIn, d_lut, imageW, imageH);
    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_lut);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaUnbindTexture(texRef); // Unbind texture memory
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    free(lut);
    return(millisecondsTransfers);
}