#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#include "CycleTimer.h"

#define N (513)
#define P (9)
#define SQRT_P (3)
#define NUM_BLOCKS (3)
#define THREADS_PER_BLOCK 1024
#define BLOCKSIZE (1024)
#define UPDIV(N, threadsPerBlock) (((N) + (threadsPerBlock) - 1) / (threadsPerBlock))


// From http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
__global__ void init(unsigned int seed, curandState_t* states) {
     curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randoms(curandState_t* states, int* numbers) {
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % N;
}

__global__ void make_sample(double* in_data,
                            double* out_data,
                            int* sample_idx,
                            int n,
                            int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    for (int i = 0; i < p; i++) {
        int row_idx = sample_idx[idx];
        out_data[idx * p + i] = in_data[row_idx * p + i];
    }
}

void bootstrap_sample(double* in_data, double** out_data, int n, int p) {
    int* device_idx;
    curandState_t* states;

    cudaMalloc((void**) &device_idx, sizeof(int) * n);
    cudaMalloc((void**) &states, sizeof(curandState_t) * n);

    init<<<N, 1>>>(time(0), states);
    randoms<<<N, 1>>>(states, device_idx);


    const int blocks = UPDIV(N, THREADS_PER_BLOCK);
    make_sample<<<blocks, THREADS_PER_BLOCK>>>(in_data, *out_data, device_idx, n, p);

    cudaFree(device_idx);
}

void bootstrap_sample(int** device_nums) {
    cudaMalloc(device_nums, sizeof(int) * N);
    curandState_t* states;
    cudaMalloc((void**) &states, sizeof(curandState_t) * N);

    init<<<N, 1>>>(time(0), states);
    randoms<<<N, 1>>>(states, *device_nums);

    cudaFree(states);
}

