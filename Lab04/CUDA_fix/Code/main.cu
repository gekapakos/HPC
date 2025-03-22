#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

bool run_cpu_gray_test(PGM_IMG img_in, char *out_filename);
bool run_GPU_gray_test(PGM_IMG img_in, char *out_filename);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    struct timespec  tv1, tv2;
    float elapsed_time_CPU;
    bool result;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);

    result = run_GPU_gray_test(img_ibuf_g, argv[2]);

    free_pgm(img_ibuf_g);

	return 0;
}

bool run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    PGM_IMG img_obuf;

    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    if (img_obuf.img == NULL)  {
        free_pgm(img_obuf);
        return(false);
    }
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
    return(true);
}

bool run_GPU_gray_test(PGM_IMG img_in, char *out_filename)
{
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_GPU(img_in);
    if (img_obuf.img == NULL)  {
        free_pgm(img_obuf);
        return(false);
    }
    write_pgm(img_obuf, out_filename);
    free_gpu_pgm(img_obuf);
    return(true);
}

bool run_gpu_gray_test(PGM_IMG img_in, char *out_filename)  {
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_GPU(img_in);
    if (img_obuf.img == NULL)  {
        return(false);
    }
    cudaDeviceSynchronize(); 
    write_pgm(img_obuf, out_filename);
    free_gpu_pgm(img_obuf);
    return(true);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    cudaMallocManaged(&result.img, result.w * result.h * sizeof(unsigned char));
        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    cudaFree(img.img);
}

void free_gpu_pgm(PGM_IMG img)
{
    cudaFree(img.img);
}