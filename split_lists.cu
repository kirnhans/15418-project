#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#include "CycleTimer.h"

#define N (513)
#define THREADS_PER_BLOCK 1024
#define UPDIV(n, threadsPerBlock) (((n) + (threadsPerBlock) - 1) / (threadsPerBlock))

__global__ void determine_direction(int* flag,
                                    int* rid,
                                    int split_value_idx,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    flag[rid[idx]] = idx > split_value_idx ? 0 : 1;
}

struct is_left {
    __host__ __device__
    bool operator()(const int &x) {
        return x == 1;
    }
};

__global__ void find_row_flag(int* flag,
                              int* flag_i,
                              int* rid,
                              int* rid_i,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    flag_i[idx] = flag[rid[idx]];
}

void split_attribute_list(double** attribute_value_list,
                          int** class_label_list,
                          int** rid_list,
                          int n,
                          int p,
                          int p_idx,
                          int split_value_idx,
                          double** right_attribute_value_list,
                          double** left_attribute_value_list,
                          int** right_class_label_list,
                          int** left_class_label_list,
                          int** right_rid_list,
                          int** left_rid_list,
                          int* right_n,
                          int* left_n) {
    //std::cout << "in split attribute list" << std::endl;
    int* flag;
    cudaMalloc((void**)&flag, sizeof(int) * n);
    const int blocks = UPDIV(N, THREADS_PER_BLOCK);

    //std::cout << "find flag array" << std::endl;
    determine_direction<<<blocks, THREADS_PER_BLOCK>>>(flag,
                                                       rid_list[p_idx],
                                                       split_value_idx,
                                                       n);

    cudaThreadSynchronize();

    //std::cout << "sort flag array by row ids" << std::endl;
    thrust::device_ptr<int> device_flag(flag);
    thrust::device_ptr<int> device_rid(rid_list[p_idx]);
    thrust::sort_by_key(device_flag, device_flag + n, device_rid);


    int* flag_i;
    cudaMalloc((void**)&flag_i, sizeof(int) * n);

    for (int i = 0; i < p; i++) {
        int* class_labels = class_label_list[i];
        double* attr_list = attribute_value_list[i];
        int* rids = rid_list[i];

        if (i == p_idx) {
            int pos = split_value_idx == n - 1 ? n - 1 : split_value_idx + 1;
            right_attribute_value_list[i] = &attr_list[pos];
            left_attribute_value_list[i] = attr_list;
            right_class_label_list[i] = &class_labels[pos];
            left_class_label_list[i] = class_labels;
            right_rid_list[i] = &rids[pos];
            left_rid_list[i] = rids;
            continue;
        }

        find_row_flag<<<blocks, THREADS_PER_BLOCK>>>(flag,
                                                     flag_i,
                                                     rid_list[p_idx],
                                                     rid_list[i],
                                                     n);

        cudaThreadSynchronize();

        thrust::device_ptr<int> device_flag_i(flag_i);
        thrust::device_ptr<double> device_attr_list(attr_list);

        /*
        int* test = new int[n];
        cudaMemcpy(test, flag_i, sizeof(int) * n, cudaMemcpyDeviceToHost);

        for (int j = 0; j < n; j++) {
            std::cout << "flag_i[" << j << "] = " << test[j] << std::endl;
        }


        delete[] test;

*/

        //std::cout << "partition attribute list" << std::endl;
        thrust::device_ptr<double> attr_ptr = thrust::stable_partition(device_attr_list,
                                                                       device_attr_list + n,
                                                                       device_flag_i,
                                                                       is_left());

        //std::cout << "partition class_labels" << std::endl;
        thrust::device_ptr<int> device_class_labels(class_labels);
        thrust::device_ptr<int> class_ptr = thrust::stable_partition(device_class_labels,
                                                                     device_class_labels + n,
                                                                     device_flag_i,
                                                                     is_left());

        //std::cout << "partition row ids" << std::endl;
        thrust::device_ptr<int> device_rids(rids);
        thrust::device_ptr<int> rid_ptr = thrust::stable_partition(device_rids,
                                                                    device_rids + n,
                                                                    device_flag_i,
                                                                    is_left());

        //std::cout << "make ptrs for left and right" << std::endl;
        int attr_pos = &attr_ptr[0] - &device_attr_list[0];
        int class_pos = &class_ptr[0] - &device_class_labels[0];
        int rid_pos = &rid_ptr[0] - &device_rids[0];

        //std::cout << "attr_pos = " << attr_pos << std::endl;
        //std::cout << "class_pos = " << class_pos << std::endl;
        //std::cout << "rid_pos = " << rid_pos << std::endl;


        right_attribute_value_list[i] = &attr_list[attr_pos];
        left_attribute_value_list[i] = attr_list;
        right_class_label_list[i] = &class_labels[class_pos];
        left_class_label_list[i] = class_labels;
        right_rid_list[i] = &rids[rid_pos];
        left_rid_list[i] = rids;
    }

    *left_n = split_value_idx + 1;
    *right_n = n - split_value_idx - 1;

    cudaFree(flag_i);
    cudaFree(flag);
}
