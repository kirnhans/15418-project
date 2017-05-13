#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

__global__ void split() {
}

void data_to_device(double* device_input_data, double* input_data, int size) {
    cudaMalloc((void **)&device_input_data, sizeof(double) * size);
    cudaMemcpy(device_input_data, input_data, sizeof(double) * size, cudaMemcpyHostToDevice);
}

void data_to_device(int* device_input_data, int* input_data, int size) {
    cudaMalloc((void **)&device_input_data, sizeof(int) * size);
    cudaMemcpy(device_input_data, input_data, sizeof(int) * size, cudaMemcpyHostToDevice);
}

void free_from_device(int* device_data) {
    cudaFree(device_data);
}

void free_from_device(double* device_data) {
    cudaFree(device_data);
}

