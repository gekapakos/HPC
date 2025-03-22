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
    extern __shared__ int private_hist[];
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    private_hist[tx] = 0;

    __syncthreads();

    for (int i = index; i < image_size; i += blockDim.x*gridDim.x)  {
        atomicAdd(&private_hist[img_in[i]], 1);
    }
    __syncthreads();

    atomicAdd(&hist_out[tx], private_hist[tx]);
}


texture<int, cudaTextureType1D, cudaReadModeElementType> lut_texture; // Bind the 1D texture
__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < imageW * imageH) {
        img_out[index] = tex1Dfetch(lut_texture, img_in[index]);
    }
}

int* lut_calculation(int * hist_in, int nbr_bin, int img_size) {
    int i, cdf, min, d, index, *lut_d;
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);

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
    for (i = 0; i < index + 1; i++) {
        lut[i] = 0;
    }

    d = img_size - min;
    for(i = index; i < nbr_bin; i++) {
        cdf += hist_in[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }
    for(i = 0; i < nbr_bin; i++) {
        if(lut[i] > 255) {
            lut[i] = 255;
        }
    }

    return lut;
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in) {
    PGM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    return result;
}

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in) {
    cudaEvent_t start_cuda, stop_cuda, start_cuda1, stop_cuda1;
    PGM_IMG img_out;
    PGM_IMG result;
    float time = 0, time_new = 0;
    int hist_h[256];
    int *hist_d;
    int seg_size = (img_in.w * img_in.h)/2;
    cudaStream_t stream[2];

    result.w = img_in.w;
    result.h = img_in.h;

    cudaMallocManaged(&img_out.img, result.w * result.h * sizeof(unsigned char));
    img_out.w = img_in.w;
    img_out.h = img_in.h;

    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventCreate(&start_cuda1);
    cudaEventCreate(&stop_cuda1);

    cudaEventRecord(start_cuda);

    cudaError_t err = cudaMalloc((void**)&hist_d, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }
    
    cudaMemset(hist_d, 0, sizeof(int) * 256);

    for (int i=0; i < 2; i++)  {
        int offset = i * seg_size;
        cudaStreamCreate(&stream[i]);

        histogramGPU<<<((seg_size)/256)+1, 256, 256*sizeof(int), stream[i] >>>(hist_d, img_in.img+offset, seg_size);

        cudaMemcpyAsync(hist_h, hist_d, 256*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&time, start_cuda, stop_cuda);

    int *lut = lut_calculation(hist_h, 256, img_out.w * img_out.h);

    cudaEventRecord(start_cuda1, 0);

    printf("\nGPU1 Execution time: %lf seconds\n", time/1000.0);
    int *lut_d;
    err = cudaMalloc((void**)&lut_d, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }

    cudaBindTexture(0, lut_texture, lut_d, 256 * sizeof(int));

    for (int i=0; i < 2; i++) {
        int offset = i * seg_size;
        cudaMemcpyAsync(lut_d, lut, 256*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        histogram_equalization_GPU<<<((seg_size)/256)+1, 256, 256*sizeof(int), stream[i]>>>(img_out.img + offset, img_in.img+offset, lut_d, img_in.w, img_in.h/2);

    }

    cudaFree(hist_d);
    cudaUnbindTexture(lut_texture);
    cudaEventRecord(stop_cuda1, 0);
    cudaEventSynchronize(stop_cuda1);
    cudaEventElapsedTime(&time_new, start_cuda1, stop_cuda1);

    time += time_new;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return img_out;
}