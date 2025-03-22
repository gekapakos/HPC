#include <stdio.h>
#include <omp.h>
#include <math.h>

int main() {
    int size = 1024;
    double array[size * size];
    double start, end;
    int j = 0;

    // Start timer
    start = omp_get_wtime();

    // Initialize the array
    for (int i = 0; i < size * size; i++)
        array[i] = 0.0;

    // Compute sin(i) + cos(j) for each element using OpenMP parallelism
    #pragma omp parallel for private(j)
    for (int i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            array[i * size + j] = sin(i) + cos(j);
        }
    }

    // End timer
    end = omp_get_wtime();

    // Print the total execution time
    printf("Total execution time: %f seconds\n", end - start);

    return 0;
}
