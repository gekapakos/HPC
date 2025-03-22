#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;

    // #pragma omp parallel for
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    // #pragma omp parallel for
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i - 1;

    // Initialize the look-up table untill
    // with zeros untill the last non-zero
    // element of the histogram.
    // #pragma omp parallel for
    for(i = 0; i < index - 1; i++)
    {
        lut[i] = 0;
    }

    d = img_size - min;

    #pragma omp parallel for reduction(+:cdf)
    for(i = index; i < nbr_bin; i ++) {
        cdf += hist_in[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }

    // Each previously calculated element of the
    // look-up table that exceeds the upper limit
    // of 255 (white), should be thresholded down
    // to 255. That is done here to avoid the if
    // statements inside the larger loop with the 
    // size of the image's width * height.
    for(i = 0; i < nbr_bin; i++) {
        if(lut[i] > 255) {
            lut[i] = 255;
        }
    }
    
    /* Get the result image */
    #pragma omp parallel for
    for(i = 0; i < img_size; i ++) {
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
}