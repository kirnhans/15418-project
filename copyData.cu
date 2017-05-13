#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

void copy_data(double* input, int n, int p) {
    int* device_input;

    cudaMalloc((void **)&device_input, sizeof(double) * n * p);
    cudaMemcpy(device_input, input, sizeof(double) * n * p, cudaMemcpyHostToDevice);
    cudaFree(device_input);

}
