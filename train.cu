#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define N 513

__global__ void split() {
}

void data_to_device(double** device_input_data, double* input_data, int size) {
    cudaMalloc(device_input_data, sizeof(double) * size);
    cudaMemcpy(*device_input_data, input_data, sizeof(double) * size, cudaMemcpyHostToDevice);
}

void data_to_device(int** device_input_data, int* input_data, int size) {
    cudaMalloc(device_input_data, sizeof(int) * size);
    cudaMemcpy(*device_input_data, input_data, sizeof(int) * size, cudaMemcpyHostToDevice);
}

void free_from_device(int* device_data) {
    cudaFree(device_data);
}

void free_from_device(double* device_data) {
    cudaFree(device_data);
}

// From http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
__global__ void init(unsigned int seed, curandState_t* states) {
     curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randoms(curandState_t* states, int* numbers) {
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % N;
}

void bootstrap_sample(int** device_nums) {
    cudaMalloc(device_nums, sizeof(int) * N);

    curandState_t* states;
    cudaMalloc((void**) &states, N * sizeof(curandState_t));

    init<<<N, 1>>>(time(0), states);
    randoms<<<N, 1>>>(states, *device_nums);

    cudaFree(states);
}
