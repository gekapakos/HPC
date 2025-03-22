#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

// Initialize the histogram's values as zero and then assign values to the histogram //
// based on if there is a value already there. Creating hist_out //
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
    int i;

    for (i = 0; i < nbr_bin; i++) {
        hist_out[i] = 0;
    }

    for (i = 0; i < img_size; i++) {
        hist_out[img_in[i]]++;
    }
}

// Create the histogram equlization by calculating a look-up table. Then //
// using the lut to assign values to the output image //
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index;

    cdf = 0;
    min = 0;
    i = 0;
    
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i-1;

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
    
    for(i = 0; i < img_size; i++) {
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
    free(lut);
}

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int imageW, int imageH) {
    extern __shared__ int private_hist[];
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Initialize the shared memory //
    // histogram to zero //
    private_hist[tx] = 0;

    __syncthreads();

    // For loop assignment of the private histogram //
    for (int i = index; i < imageH*imageW; i += blockDim.x*gridDim.x) {
        atomicAdd(&private_hist[img_in[i]], 1);
    }   

    __syncthreads();

    // Assign the value back to the global memory //
    atomicAdd(&hist_out[tx], private_hist[tx]);
}

// Store in the constant memory //
// the lut table to access from there //
__constant__ int constant_lut[256] ;
__global__ void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (index < imageW * imageH)  {
        img_out[index] = constant_lut[img_in[index]];
    }
}

// This is a function that first calculates the LUT vector //
// from the device and then calls the equalization function from the GPU //
// In the end it returns the time spent calculating them //
int lut_calculation(unsigned char * img_out, unsigned char * img_in, int * hist_in, int imageW, int imageH, int nbr_bin, unsigned char * img_in_d) {
    int i, cdf, min, d, index, *d_lut;
    float time = 0;
    cudaEvent_t start_cuda, stop_cuda;
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int img_size = imageW * imageH;

    // Create Cuda Events
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    cdf = 0;
    min = 0;
    i = 0;
    
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i-1;

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

    // Start Recording //
    cudaEventRecord(start_cuda, 0);

    // Allocate memory for the device LUT //
    cudaMalloc((void **)&d_lut, sizeof(int)*nbr_bin);

    // Copy the context from Host to Device //
    // Initializing the device's LUT with the //
    // host's LUT // 
    cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);  // Copy data from host to device

    // Call the Histogram Equalization for the gpu //
    histogram_equalization_gpu<<<(img_size/256)+1, 256>>>(img_out, img_in_d, d_lut, imageW, imageH);

    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch error: %s\n", cudaGetErrorString(err));
    }

    // Free the devices //
    cudaFree(d_lut);

    // Stop recording //
    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);

    // Record the time spent //
    cudaEventElapsedTime(&time, start_cuda, stop_cuda);

    // Free the host //
    free(lut);

    // Return the time spent in the device //
    return(time);
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
    PGM_IMG img_out;
    PGM_IMG result;
    float time_new = 0, time;
    int hist_h[256];
    int *hist_d;
    unsigned char *img_in_d;
    cudaEvent_t startCuda, stopCuda;
    cudaError_t err;

    // Create Cuda Events to calculate the //
    // time //
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    // Set the results variables values //
    result.w = img_in.w;
    result.h = img_in.h;

    // Unified memory call for the input image //
    cudaMallocManaged(&img_out.img, result.w * result.h * sizeof(unsigned char));

    // Set the img_out variables //
    img_out.w = img_in.w;
    img_out.h = img_in.h;

    // Start Recording //
    cudaEventRecord(startCuda);

    // Allocate memory in the device for the histogram //
    err = cudaMalloc((void**)&hist_d, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc error: %s\n", cudaGetErrorString(err));
        if (img_out.img) cudaFree(img_out.img);
        return(img_out);
    }
    
    // Initialize the histogram with zeros //
    cudaMemset(hist_d, 0, sizeof(int) * 256);
    
    // Call the device's histogram function //
    histogram_gpu<<<((img_out.h*img_out.w)/256)+1, 256, 256*sizeof(int) >>>(hist_d, img_in.img, img_out.w, img_out.h);

    // Check if everything is ok with //
    // the above kernel function //
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy the values from the cpu's histogram to //
    // the gpu's histogram //
    err = cudaMemcpy(hist_h, hist_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device

    // Stop Recording //
    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    // Calculate the time passed //
    cudaEventElapsedTime(&time_new, startCuda, stopCuda);

    // Call the function, that returns the time //
    // spent by the GPU operations //
    time = lut_calculation(img_out.img, img_in.img, hist_h, img_out.w, img_out.h, 256, img_in.img);

    // Update the total time spent //
    time += time_new;

    // Calculate the time for the last Free //
    // in the code //
    cudaEventRecord(startCuda, 0);

    cudaFree(hist_d);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&time_new, startCuda, stopCuda);

    time += time_new;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return img_out;
}