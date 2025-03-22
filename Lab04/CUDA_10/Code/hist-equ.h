#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

extern texture<int, cudaTextureType1D, cudaReadModeElementType> texRef; // Bind the 1D texture

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);
void free_gpu_pgm(PGM_IMG img);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);
int* histogram_equalization_prep(int * hist_in, int nbr_bin, int img_size);
__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int image_size);
__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);
PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in);

#endif