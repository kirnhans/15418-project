#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "DecisionTreeRepr.h"

#define UPDIV(N, threadsPerBlock) ((N + threadsPerBlock -1)/threadsPerBlock)

__global__ void
insertKernel (struct dec_tree t, int elem) {
int done=0;
int cur_node = 0;
while (!done) {
if (!dec_tree[cur_node].full) {
dec_tree[cur_node].value = elem;
done = 1;
}
 else {
if (dec_tree[cur_node].value < elem) {
cur_node = 2 * cur_node + 1; //right child
}
 else {
cur_node = 2 * cur_node; //left child
}
}
