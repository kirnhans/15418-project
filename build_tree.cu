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

//requires 0 < m <= num columns
//maybe we should make a dataset struct?
__global__ void
buildLevel(struct dec_tree t, int cur_node, int** dataset, int size_dataset, int num_cols) {

    //-measure impurity - do we need to split at all?
    //-entropy = sum -(pi*log2(pi)) where pi = # of prediction i / total # of predictions or Gini impurity
    double impurity = 0;  //leave this one to Sally, I don't know the difference

    if (impurity > IMPURITY_THRESHOLD) {
	// -randomly choose m features without replacement
	int *random_index = new int[m];
	double *impurity = new double[m];
	double max_impurity = 0;
	double max_index = 0;
	for (int i = 0; i < m; i++) {

	    int read_index;
	    do {
		read_index = rand() % len;
	    }
	    while (contains(random_index, read_index));
	    random_index[i]=read_index;

	    //measure impurity per feature we could split on
	    impurity[i] = 0;//stats_magic_that_sally_knows

	    //get max impurity -> best feature to split on
	    if (impurity[i] > max_impurity) {
		max_impurity = impurity[i];
		max_index = i;
	    }
	}
	//split on best feature
	//create new subsets of data for each child node
	int split_val = 0; //more stats magic?
	t[cur_node] = split_val;

	//how do we know how much to allocate? Magic, please
	int **left_dataset = new int*[size_dataset];
	int **right_dataset = new int*[size_dataset];
	int left_data_counter = 0;
	int right_data_counter = 0;
	for (int i = 0; i < size_dataset; i++) {

	    //if splitting feature > split median
	    if (dataset[i][max_index] > split_val) {
	    //copy over datasets to children
		right_dataset[right_data_counter] = new int[num_cols];
		memcpy(dataset[i], right_dataset[right_data_counter],
		       sizeof(int) * num_cols);
		right_data_counter++;
	    }
	    else {
		left_dataset[left_data_counter] = new int[num_cols];
		memcpy(dataset[i], left_dataset[left_data_counter],
		       sizeof(int) * num_cols);
		left_data_counter++;
	    }
	}

	//-repeat process at children nodes
	//could parallelize this
	build_tree(t, left_child(cur_node), left_data_set, left_data_counter, num_cols);
	build_tree(t, right_child(cur_node), right_data_set, right_data_counter, num_cols);
    //This is good for CUDA because it's divergent

}
