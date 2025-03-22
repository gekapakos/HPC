/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

// Modifcation 4 for HW: Padding on CPU and GPU //

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 

 

////////////////////////////////////////////////////////////////////
// Reference row conon filter
///////////////////////
void convolutionRowCPUPadded(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = (x+filterR) + k;

        sum += h_Src[(y + filterR) * (imageW + filterR*2) + d] * h_Filter[filterR - k];

        h_Dst[(y+filterR) * (imageW + filterR*2) + (x+filterR)] = sum;
      }
    }
  }
        
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int k;
 
  row = row + filterR;
  col = col + filterR;

  float sum = 0.0;

  for (k = -filterR; k <= filterR; k++) {
    int d = (col) + k;
    sum += d_Src[(row) * (imageW + filterR * 2) + d] * d_Filter[filterR - k];
  }
  d_Dst[(row) * (imageW + filterR*2) + (col)] = sum;
  printf("%f\n", d_Dst[(row) * (imageW + filterR*2) + (col)]);
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPUPadded(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = (y+filterR) + k;

        sum += h_Src[d * (imageW + filterR*2) + (x+filterR)] * h_Filter[filterR - k];
 
        h_Dst[(y+filterR) * (imageW + filterR*2) + (x+filterR)] = sum;
      }
    }
  }
    
}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
  int imageW, int imageH, int filterR) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int k;

  row = row + filterR;
  col = col + filterR;

  float sum = 0.0;
  for (k = -filterR; k <= filterR; k++) {
    int d = (row) + k;

    sum += d_Src[d * (imageW + filterR * 2) + (col)] * d_Filter[filterR - k];
    
  }
  d_Dst[(row) * (imageW + filterR*2) + (col)] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    // *h_Buffer,
    // *h_OutputCPU,
    *h_InputPadded,
    *h_BufferPadded,
    *h_OutputCPUPadded;

    float
    *d_Filter = NULL,
    *d_Input = NULL,
    *d_Buffer = NULL,
    *d_OutputGPU = NULL;


    int imageW;
    int imageH;
    unsigned int i, j;

    // float accuracy; // the accuracy of the program //

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
    h_InputPadded     = (float *)malloc((imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float));
    h_BufferPadded    = (float *)malloc((imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float));
    h_OutputCPUPadded = (float *)malloc((imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float));

    h_Filter    = (float *)malloc((FILTER_LENGTH) * sizeof(float));
    h_Input     = (float *)malloc((imageW) * (imageH) * sizeof(float));
    // h_Buffer    = (float *)malloc((imageW) * (imageH) * sizeof(float));
    // h_OutputCPU = (float *)malloc((imageW) * (imageH) * sizeof(float));

    /* Allocate memory for the 2nd step alteration */
    h_OutputCPU_from_GPU = (float *)malloc((imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float));
    if (h_OutputCPU_from_GPU == NULL) {
        printf("Error allocating memory for h_OutputCPU_from_GPU\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate Memory for the device(GPU) */
    // Check allocation of device memory
    if (cudaMalloc((void**) &d_Filter, FILTER_LENGTH * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Filter on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Input, (imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Input on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_Buffer, (imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_Buffer on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMalloc((void**) &d_OutputGPU, (imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float)) != cudaSuccess) {
      printf("Error allocating memory for d_OutputGPU on the device: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < (imageW) * (imageH); i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    
    for (i = 0; i < (imageH); i++) {
    for(j = 0; j < (imageW); j++) { 
      h_InputPadded[(i+filter_radius)*(imageW+filter_radius*2) + (j + filter_radius)] = h_Input[i*imageW + j];
    }

  for (i = 0; i < (imageH + filter_radius * 2); i++) {
      for(j = 0; j < (imageW + filter_radius * 2); j++) {
        // printf("%d: %f\n", (i*(imageW +2* filter_radius) + j), h_InputPadded[i * (imageW + filter_radius*2) + j]);
      }
  }
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    /* Copy Memory from host to device */
    err = cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);
    if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    }

    elapsed_time_GPU = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    cudaMemcpy(d_Input, h_InputPadded, (imageW + filter_radius*2) * (imageH + filter_radius*2) * sizeof(float), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);
    if (err != cudaSuccess) {
      printf("Error allocating memory on the device: %s\n", cudaGetErrorString(err));
    }

    elapsed_time_GPU += (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPUPadded(h_BufferPadded, h_InputPadded, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPUPadded(h_OutputCPUPadded, h_BufferPadded, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    // Setup the execution configuration
    const int TileWidth = 32;

    dim3 dimGrid(imageW / TileWidth, imageW / TileWidth);
    dim3 dimBlock(TileWidth, TileWidth);

    printf("GPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    cudaMemcpy(h_OutputCPU_from_GPU, d_OutputGPU, (imageW + filter_radius*2) * (imageH + filter_radius*2) *sizeof(float), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    elapsed_time_GPU += (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_nsec - tv3.tv_nsec) / 1e9;  

    int find_diff = 0;
    float diff_result = 0;

    // in order to make prints //
    cudaDeviceSynchronize();
    // Compare the two outputs:
    diff = 0;
    for(i = 0; i < (imageH + filter_radius*2) * (imageW + filter_radius*2); i++)
    {

      diff = h_OutputCPUPadded[i] - h_OutputCPU_from_GPU[i];
      diff = ABS(diff);
      if(diff != 0)
      {
        diff_result += diff;
        // printf("(%f, %f) --> %f\n", h_OutputCPUPadded[i], h_OutputCPU_from_GPU[i], diff);
        find_diff++;
      }
      if(diff > max)
      {
        max = diff;
      }
    }

    diff_result = diff_result / find_diff;

    // free all the allocated memory
    if(h_Input) free(h_Input);
    if(h_Filter) free(h_Filter);

    if(h_OutputCPUPadded) free(h_OutputCPUPadded);
    if(h_BufferPadded) free(h_BufferPadded);
    if(h_InputPadded) free(h_InputPadded);

    // free device (GPU) memory
    if(d_OutputGPU) cudaFree(d_OutputGPU);
    if(d_Buffer) cudaFree(d_Buffer);
    if(d_Input) cudaFree(d_Input);
    if(d_Filter) cudaFree(d_Filter);

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
