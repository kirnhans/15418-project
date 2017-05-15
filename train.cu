#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#include "CycleTimer.h"

#include "macros.h"

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

__global__ void init_mask(int* mask, int n, int val) {
    if (blockIdx.x >= n) {
        return;
    }

    mask[blockIdx.x] = val;
}

void init_data_mask(int** data_mask, int n, int val) {
    cudaMalloc(data_mask, sizeof(int) * n);
    init_mask<<<N, 1>>>(*data_mask, n, val);
}

__global__ void kernel_try_splits(int* data_mask, int* data_idx, double* data,
                                  int* label, int n, int p, double* gini_vals,
                                  double* split_vals, int* left_count, int* sizes) {
    int p_idx = blockIdx.x;
    double min_gini = 1.0;
    double value = 0.0;
    int left_totals = 0;
    int size = 0;

    for (int i = 0; i < n; i++) {
        int data_i = data_idx[i];
        if (!data_mask[data_i]) {
            continue;
        }

        double p_value = data[data_i * p + p_idx];
        int left_count = 0;
        int right_count = 0;
        int left_1 = 0;
        int right_1 = 0;

        for (int j = 0; j < n; j++) {
            int data_j = data_idx[j];
            if (!data_mask[data_j]) {
                continue;
            }

            double data_value = data[data_j * p + p_idx];

            if (data_value >= p_value) {
                right_count++;
                right_1 += label[data_j];
            } else {
                left_count++;
                left_1 += label[data_j];
            }
        }

        double p_left_1 = left_count > 0 ?
                          double(left_1) / double(left_count) : 0;
        double p_left_0 = left_count > 0 ?
                          double(left_count - left_1) / double(left_count) : 0;
        double p_right_1 = right_count > 0 ?
                           double(right_1) / double(right_count) : 0;
        double p_right_0 = right_count > 0 ?
                           double(right_count - right_1) / double(right_count) : 0;

        double gini = p_left_1 * (1 - p_left_1) +
            p_left_0 * (1 - p_left_0) +
            p_right_1 * (1 - p_right_1) +
            p_right_0 * (1 - p_right_0);

        if (gini < min_gini) {
            min_gini = gini;
            value = p_value;
            left_totals = left_count;
            size = left_count + right_count;
        }
    }
    gini_vals[p_idx] = min_gini;
    split_vals[p_idx] = value;
    left_count[p_idx] = left_totals;
    sizes[p_idx] = size;
}

__global__ void make_split_mask(int* data_mask,
                           int* data_idx,
                           double* data,
                           int n,
                           int p,
                           int p_idx,
                           double split_val,
                           int* left_mask,
                           int* right_mask) {

    if (blockIdx.x >= n) {
        return;
    }

    int data_i = data_idx[blockIdx.x];

    double data_val = data[data_i * p + p_idx];
    left_mask[blockIdx.x] = data_mask[data_i] && data_val < split_val;
    right_mask[blockIdx.x] = data_mask[data_i] && data_val >= split_val;
}

void find_split(int* data_mask,
                int* data_idx,
                double* data,
                int* label,
                int n,
                int p,
                int** left_mask,
                int** right_mask,
                int* split_p_idx,
                double* split_p_val,
                double* split_gini,
                int* split_left_count,
                int* split_right_count) {
    double* device_gini_vals;
    double* device_split_vals;
    int* device_left_count;
    int* device_sizes;

    cudaMalloc((void**) &device_gini_vals, p * sizeof(double));
    cudaMalloc((void**) &device_split_vals, p * sizeof(double));
    cudaMalloc((void**) &device_left_count, p * sizeof(int));
    cudaMalloc((void**) &device_sizes, p * sizeof(int));

    cudaMalloc(left_mask, n * sizeof(int));
    cudaMalloc(right_mask, n * sizeof(int));

    double* gini_vals = new double[p];
    double* split_vals = new double[p];
    int* left_count = new int[p];
    int* sizes = new int[p];

    // TODO only use a random subset of variables instead of all

    // TODO add more threads per block to help handle data in parallel.
    kernel_try_splits<<<P, 1>>>(data_mask,
                                data_idx,
                                data,
                                label,
                                n,
                                p,
                                device_gini_vals,
                                device_split_vals,
                                device_left_count,
                                device_sizes);

    cudaMemcpy(gini_vals, device_gini_vals, p * sizeof(double),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(split_vals, device_split_vals, p * sizeof(double),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(left_count, device_left_count, p * sizeof(int),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(sizes, device_sizes, p * sizeof(int),
            cudaMemcpyDeviceToHost);

    // TODO use kernel to find minimum instead
    int p_idx = -1;
    double min_gini = 1.0;
    double split_value = 0.0;
    int left_total = 0;
    int node_size = 0;

    for (int i = 0; i < p; i++) {
        if (gini_vals[i] < min_gini) {
            min_gini = gini_vals[i];
            split_value = split_vals[i];
            p_idx = i;
            left_total = left_count[i];
            node_size = sizes[i];
        }
    }

    *split_p_idx = p_idx;
    *split_p_val = split_value;
    *split_gini = min_gini;
    *split_left_count = left_total;
    *split_right_count = node_size - left_total;

    make_split_mask<<<N, 1>>>(data_mask,
                              data_idx,
                              data,
                              n,
                              p,
                              p_idx,
                              split_value,
                              *left_mask,
                              *right_mask);

    cudaFree(device_gini_vals);
    cudaFree(device_split_vals);
    cudaFree(device_left_count);
    cudaFree(device_sizes);

    delete[] gini_vals;
    delete[] split_vals;
    delete[] left_count;
    delete[] sizes;
}
