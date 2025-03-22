/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter for CPU (Unchanged)
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter for GPU with only threads (1st Approach)
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int k;
  int idx_x = threadIdx.x;
  int idx_y = threadIdx.y;
  
  float sum = 0;

  for (k = -filterR; k <= filterR; k++) {
    int d = idx_x + k;

    if (d >= 0 && d < imageW) {
      sum += h_Src[idx_y * imageW + d] * h_Filter[filterR - k];
    }     

    h_Dst[idx_y * imageW + idx_x] = sum;
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter GPU with threads (1st Approach)
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int k;
  int idx_x = threadIdx.x;
  int idx_y = threadIdx.y;
  float sum = 0;

  for (k = -filterR; k <= filterR; k++) {
    int d = idx_y + k;

    if (d >= 0 && d < imageH) {
      sum += h_Src[d * imageW + idx_x] * h_Filter[filterR - k];
    }   
 
    h_Dst[idx_y * imageW + idx_x] = sum;
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;

    float
    *d_Filter = NULL,
    *d_Input = NULL,
    *d_Buffer = NULL,
    *d_OutputGPU = NULL;

    int imageW;
    int imageH;
    unsigned int i;

    struct timespec  tv1, tv2, tv3, tv4;
    float elapsed_time_CPU, elapsed_time_GPU;
    float max = 0.0, diff = 0.0;
    cudaError_t err;
    float *h_OutputCPU_from_GPU;

    printf("Enter filter radius : ");
    scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));

    /* Allocate memory for the 2nd step alteration */
    h_OutputCPU_from_GPU = (float *)malloc(imageW * imageH * sizeof(float));
    if (h_OutputCPU_from_GPU == NULL) {
        printf("Error allocating memory for h_OutputCPU_from_GPU\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate Memory for the device(GPU) */
    // Check allocation of device memory
    if (cudaMalloc((void**) &d_Filter, FILTER_LENGTH * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Filter on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Input on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Buffer on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_OutputGPU on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    /* Copy Memory from host to device */
    err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    }

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    // Setup the execution configuration
    dim3 dimGrid(1, 1);
    dim3 dimBlock(imageW, imageW);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    

    cudaMemcpy(h_Buffer, d_Buffer, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_OutputCPU_from_GPU, d_OutputGPU, imageW * imageH *sizeof(float), cudaMemcpyDeviceToHost);


    // in order to make prints //
    cudaDeviceSynchronize();
    // Compare the two outputs:
    diff = 0;
    for(i = 0; i < imageH * imageW; i++)
    {

      diff = h_OutputCPU[i] - h_OutputCPU_from_GPU[i];
      diff = ABS(diff);
      if(diff != 0)
      {
        // printf("(%f, %f) --> %f\n", h_OutputCPU[i], h_OutputCPU_from_GPU[i], diff);
      }
      if(diff > max)
        max = diff;
    }

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // free device (GPU) memory
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
    cudaFree(d_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();

    // Calculate the elapsed CPU time in seconds
    elapsed_time_CPU = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_nsec - tv1.tv_nsec) / 1e9;

    // Print the execution time
    printf("CPU Execution time: %lf seconds\n", elapsed_time_CPU);

    // Calculate the elapsed GPU time in seconds
    elapsed_time_GPU = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;

    // Print the execution time
    printf("GPU Execution time: %lf seconds\n", elapsed_time_GPU);

    printf("The accuracy based on the worse performing difference is: %lf\n", 100-max);

    if(max > accuracy)
    {
      printf("The program is not correct\n");
    }
    else
    {
      printf("The program is correct\n");
    }

    return 0;
}