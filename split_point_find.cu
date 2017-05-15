#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#include "CycleTimer.h"

#include "macros.h"

__global__ void check_if_split(double* attribute_values,
                          int n,
                          int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    } else if (idx == n - 1) {
        result[idx] = 1;
        //printf("idx: %d, setting results[%d] = 1\n", idx, idx);
    } else {
        result[idx] = attribute_values[idx] != attribute_values[idx + 1] ? 1 : 0;
        //printf("idx: %d, setting results[%d] = %d\n", idx, result[idx]);
    }
}

__global__ void fill_buffer(int* buffer,
                            int* buffer_idx,
                            int* c,
                            int* flag,
                            int* addr,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    if (flag[idx] == 1) {
        buffer[addr[idx] - 1] = c[idx];
        buffer_idx[addr[idx] - 1] = idx;
    }
}

void compact(int* c,
             int* flag,
             int* addr,
             int* buffer,
             int* buffer_idx,
             int size,
             int n) {
    const int blocks = UPDIV(N, THREADS_PER_BLOCK);
    fill_buffer<<<blocks, THREADS_PER_BLOCK>>>(buffer,
                                               buffer_idx,
                                               c,
                                               flag,
                                               addr,
                                               n);
    cudaThreadSynchronize();
}

__global__ void calculate_gini(int* buffer,
                               int* buffer_idx,
                               int size,
                               int n,
                               double* values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    // Note: left count includes the current value because left is <=
    int left_count = buffer_idx[idx] + 1;
    int right_count = n - left_count;
    int left_1 = buffer[idx];
    int right_1 = buffer[size - 1] - left_1;

    double p_left_1 = left_count > 0 ?
                      double(left_1) / double(left_count) : 0;
    double p_right_1 = right_count > 0 ?
                       double(right_1) / double(right_count) : 0;

    double gini = 2 * (p_left_1 * (1 - p_left_1)) +
                  2 * (p_right_1 * (1 - p_right_1));

    values[idx] = gini;
}

void find_split(double** attribute_value_list,
                int** class_label_list,
                int** rid_list,
                int n,
                int p,
                int* best_attr,
                int* split_idx,
                double* best_gini) {
    //std::cout << "in find split" << std::endl;

    double min_gini = 1.0;
    int p_idx = -1;
    int split_value_idx = -1;

    int* c_i;
    cudaMalloc((void**)&c_i, n * sizeof(int));
    thrust::device_ptr<int> device_ci(c_i);

    int* is_split_pt;
    cudaMalloc((void**)&is_split_pt, n * sizeof(int));

    int* addr_i;
    cudaMalloc((void**)&addr_i, n * sizeof(int));
    thrust::device_ptr<int> device_addr_i(addr_i);

    //std::cout << "begin loop over params" << std::endl;
    for (int i = 0; i < p; i++) {
        thrust::device_ptr<int> device_class_label_list(class_label_list[i]);
        thrust::inclusive_scan(device_class_label_list,
                               device_class_label_list + n,
                               device_ci);

        //std::cout << "check if split kernel" << std::endl;
        const int blocks = UPDIV(N, THREADS_PER_BLOCK);
        check_if_split<<<blocks, THREADS_PER_BLOCK>>>(attribute_value_list[i],
                                                      n,
                                                      is_split_pt);

        cudaThreadSynchronize();

        //std::cout << "scan is_split" << std::endl;

        /*
        int* test = new int[n];
        cudaMemcpy(test, is_split_pt, n * sizeof(int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < n; j++) {
            std::cout << "is_split_pt[" << j << "] = " << test[j] << std::endl;
        }
        delete[] test;
*/


        thrust::device_ptr<int> device_is_split_pt(is_split_pt);
        thrust::inclusive_scan(device_is_split_pt,
                               device_is_split_pt + n,
                               device_addr_i);

        //std::cout << "get count" << std::endl;
        int* buffer;
        int* buffer_idx;
        int size = thrust::count(device_is_split_pt, device_is_split_pt + n, 1);
        cudaMalloc((void**)&buffer, size * sizeof(int));
        cudaMalloc((void**)&buffer_idx, size * sizeof(int));

        //std::cout << "compact" << std::endl;
        compact(c_i,
                is_split_pt,
                addr_i,
                buffer,
                buffer_idx,
                size,
                n);

        //std::cout << "gini index calculate" << std::endl;
        double* values;
        cudaMalloc((void**)&values, size * sizeof(double));
        calculate_gini<<<blocks, THREADS_PER_BLOCK>>>(buffer,
                                                      buffer_idx,
                                                      size,
                                                      n,
                                                      values);

        cudaThreadSynchronize();


        //std::cout << "find min gini" << std::endl;
        thrust::device_ptr<double> device_values(values);
        thrust::device_ptr<double> min_ptr = thrust::min_element(device_values, device_values + size);

        double min_value = min_ptr[0];
        if (min_value < min_gini) {
            min_gini = min_value;
            p_idx = i;
            split_value_idx = &min_ptr[0] - &device_values[0];
        }

        cudaFree(values);
        cudaFree(buffer);
        cudaFree(buffer_idx);
    }

    cudaFree(c_i);
    cudaFree(is_split_pt);
    cudaFree(addr_i);


    *best_attr = p_idx;
    *split_idx = split_value_idx;
    *best_gini = min_gini;
}
