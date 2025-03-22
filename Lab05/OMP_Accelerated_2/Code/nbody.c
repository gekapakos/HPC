#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */

typedef struct { 
  float x, y, z, vx, vy, vz; 
} Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  float dx;
  float dy;
  float dz;
  float distSqr;
  float invDist;
  float invDist3;
  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
  int j;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    Fx = 0.0f; Fy = 0.0f; Fz = 0.0f;
    for (j = 0; j < n; j++) {
      dx = p[j].x - p[i].x;
      dy = p[j].y - p[i].y;
      dz = p[j].z - p[i].z;
      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // Compute interbody forces

    // Update positions
    for (int i = 0; i < nBodies; i++) {
      p[i].x += p[i].vx * dt;
      p[i].y += p[i].vy * dt;
      p[i].z += p[i].vz * dt;
    }

    // Write positions and velocities to file during the first iteration
    if (iter == 1) {
      FILE *outputFile = fopen("OMP_Accelerated.txt", "w");
      if (outputFile != NULL) {
        for (int i = 0; i < nBodies; i++) {
          fprintf(outputFile, "Body %d: x=%.6f, y=%.6f, z=%.6f\n",
                  i, p[i].x, p[i].y, p[i].z);
        }
        fclose(outputFile);
      } else {
        fprintf(stderr, "Error: Could not open output file.\n");
      }
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    printf("Iteration %d: %e seconds\n", iter, tElapsed);
  }

  double avgTime = totalTime / (double)(nIters - 1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  free(buf);
}