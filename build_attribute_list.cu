#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "macros.h"

// Use this function to allocate memory on device for all the attribute lists
void init_attribute_list_memory(double** attribute_value_list,
                                int** class_label_list,
                                int** rid_list,
                                int n) {
    cudaMalloc(attribute_value_list, n * sizeof(double));
    cudaMalloc(class_label_list, n * sizeof(int));
    cudaMalloc(rid_list, n * sizeof(int));
}

__global__ void kernel_data_to_attribute_lists(double* data,
                           int* labels,
                           int n,
                           int p,
                           int p_idx,
                           double* attribute_value_list,
                           int* class_label_list,
                           int* rid_list) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    attribute_value_list[idx] = data[idx * p + p_idx];
    class_label_list[idx] = labels[idx];
    rid_list[idx] = idx;
}

void build_attribute_lists(double* data,
                           int* labels,
                           int n,
                           int p,
                           double** attribute_value_list,
                           int** class_label_list,
                           int** rid_list) {

    const int blocks = UPDIV(N, THREADS_PER_BLOCK);

    for (int i = 0; i < p; i++) {
        kernel_data_to_attribute_lists<<<blocks, THREADS_PER_BLOCK>>>(data,
                                                                      labels,
                                                                      n,
                                                                      p,
                                                                      i,
                                                                      attribute_value_list[i],
                                                                      class_label_list[i],
                                                                      rid_list[i]);
    }
    cudaThreadSynchronize();

    for (int i = 0; i < p; i++) {
        double* device_attribute_value = attribute_value_list[i];
        int* device_rid = rid_list[i];
        int* device_class_label = class_label_list[i];

        thrust::device_ptr<double> thrust_attribute_value(device_attribute_value);
        thrust::device_ptr<int> thrust_rid(device_rid);
        thrust::device_ptr<int> thrust_class_label(device_class_label);

        thrust::stable_sort_by_key(thrust_rid,
                                   thrust_rid + n,
                                   thrust_attribute_value);

        thrust::stable_sort_by_key(thrust_class_label,
                                   thrust_class_label + n,
                                   thrust_attribute_value);

        thrust::stable_sort(thrust_attribute_value,
                            thrust_attribute_value + n);
    }
}
