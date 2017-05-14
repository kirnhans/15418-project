#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define N (513)
#define THREADS_PER_BLOCK 1024
#define UPDIV(n, threadsPerBlock) ((n + threadsPerBlock - 1) / threadsPerBlock)

// Use this function to allocate memory on device for all the attribute lists
void init_attribute_list_memory(double*** attribute_value_list,
                                int*** class_label_list,
                                int*** rid_list,
                                int n,
                                int p) {
    for (int i = 0; i < p; i++) {
        cudaMalloc(attribute_value_list[i], n * sizeof(double));
        cudaMalloc(class_label_list[i], n * sizeof(int));
        cudaMalloc(rid_list[i], n * sizeof(int));
    }
}

// Use this function to copy over the data to device
void copy_data_to_device(double** device_input_data, double* input_data, int size) {
    cudaMalloc(device_input_data, sizeof(double) * size);
    cudaMemcpy(*device_input_data,
               input_data,
               sizeof(double) * size,
               cudaMemcpyHostToDevice);
}

// Use this function to copy over the class labels to device
void copy_data_to_device(int** device_input_data, int* input_data, int size) {
    cudaMalloc(device_input_data, sizeof(int) * size);
    cudaMemcpy(*device_input_data,
               input_data,
               sizeof(int) * size,
               cudaMemcpyHostToDevice);
}


__global__ void kernel_data_to_attribute_lists(double* data,
                           int* labels,
                           int n,
                           int p,
                           double** attribute_value_list,
                           int** class_label_list,
                           int** rid_list) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    for (int i = 0; i < p; i++) {
        attribute_value_list[i][idx] = data[idx * p + i];
        class_label_list[i][idx] = labels[idx];
        rid_list[i][idx] = idx;
    }
}


void build_attribute_lists(double* data,
                           int* labels,
                           int n,
                           int p,
                           double** attribute_value_list,
                           int** class_label_list,
                           int** rid_list) {

    const int blocks = UPDIV(N, THREADS_PER_BLOCK);
    kernel_data_to_attribute_lists<<<blocks, THREADS_PER_BLOCK>>>(data,
                                                                  labels,
                                                                  n,
                                                                  p,
                                                                  attribute_value_list,
                                                                  class_label_list,
                                                                  rid_list);
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
