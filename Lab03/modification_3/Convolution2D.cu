/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/
// Modifcation 3 for HW: Change floats to doubles //

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter for CPU (Unchanged)
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

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
__global__ void convolutionRowGPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) 
{

  int k;
  // int idx_x = threadIdx.x;
  // int idx_y = threadIdx.y;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
                      
  // for (y = 0; y < imageH; y++) {
  //   for (x = 0; x < imageW; x++) {
  double sum = 0;

  for (k = -filterR; k <= filterR; k++) {
    int d = col + k;

    if (d >= 0 && d < imageW) {
      sum += h_Src[row * imageW + d] * h_Filter[filterR - k];
    }     

    h_Dst[row * imageW + col] = sum;
  }
  //   }
  // }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

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
__global__ void convolutionColumnGPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int k;
  // int idx_x = threadIdx.x;
  // int idx_y = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // for (y = 0; y < imageH; y++) {
  //   for (x = 0; x < imageW; x++) {
  double sum = 0;

  for (k = -filterR; k <= filterR; k++) {
    int d = row + k;

    if (d >= 0 && d < imageH) {
      sum += h_Src[d * imageW + col] * h_Filter[filterR - k];
    }   
 
    h_Dst[row * imageW + col] = sum;
    //   }
    // }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;

    double
    *d_Filter = NULL,
    *d_Input = NULL,
    *d_Buffer = NULL,
    *d_OutputGPU = NULL;

    int imageW;
    int imageH;
    unsigned int i;

    float accuracy;

    struct timespec  tv1, tv2, tv3, tv4;
    double elapsed_time_CPU, elapsed_time_GPU;
    double max = 0.0, diff = 0.0;
    cudaError_t err;
    double *h_OutputCPU_from_GPU;

    // Accuracy from user stdin //
    printf("Enter Accuracy : ");
    scanf("%f", &accuracy);

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
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));

    /* Allocate memory for the 2nd step alteration */
    h_OutputCPU_from_GPU = (double *)malloc(imageW * imageH * sizeof(double));
    if (h_OutputCPU_from_GPU == NULL) {
        printf("Error allocating memory for h_OutputCPU_from_GPU\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate Memory for the device(GPU) */
    // Check allocation of device memory
    if (cudaMalloc((void**) &d_Filter, FILTER_LENGTH * sizeof(double)) != cudaSuccess) {
      printf("Error allocating memory for d_Filter on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(double)) != cudaSuccess) {
      printf("Error allocating memory for d_Input on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(double)) != cudaSuccess) {
      printf("Error allocating memory for d_Buffer on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(double)) != cudaSuccess) {
      printf("Error allocating memory for d_OutputGPU on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }

  clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    /* Copy Memory from host to device */
    err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);
    if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    }
    elapsed_time_GPU = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;

    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(double), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);
        if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
        }

    elapsed_time_GPU = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    // Setup the execution configuration
    const int TileWidth = 32;

    dim3 dimGrid(imageW / TileWidth, imageW / TileWidth);
    dim3 dimBlock(TileWidth, TileWidth);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    cudaMemcpy(h_Buffer, d_Buffer, imageW * imageH * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_OutputCPU_from_GPU, d_OutputGPU, imageW * imageH *sizeof(double), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    elapsed_time_GPU = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;

    int find_diff = 0;
    double diff_result = 0;

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
        diff_result += diff;
        // printf("(%f, %f) --> %f\n", h_OutputCPU[i], h_OutputCPU_from_GPU[i], diff);
        find_diff++;
      }
      if(diff > max)
        max = diff;
    }

    diff_result = diff_result / find_diff;

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
    printf("The accuracy based on the average error performing difference is: %lf\n", 100-diff_result);

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