#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#include "DecisionTree.h"

void data_to_device(double** device_input_data, double* input_data, int size);
void data_to_device(int** device_input_data, int* input_data, int size);
void free_from_device(int* device_data);
void free_from_device(double* device_data);

void bootstrap_sample(double* in_data,
                      int* in_label,
                      double** out_data,
                      int** out_label,
                      int n,
                      int p);

void init_attribute_list_memory(double** attribute_value_list,
                                int** class_label_list,
                                int** rid_list,
                                int n);

void build_attribute_lists(double* data,
                           int* labels,
                           int n,
                           int p,
                           double** attribute_value_list,
                           int** class_label_list,
                           int** rid_list);

void find_split(double** attribute_value_list,
                int** class_label_list,
                int** rid_list,
                int n,
                int p,
                int* best_attr,
                int* split_idx,
                double* best_gini);


DecisionTree::DecisionTree(double* train_data, int* train_y, int n, int p) :
                                                       train_data(train_data),
                                                       train_y(train_y),
                                                       n(n),
                                                       p(p),
                                                       mtry(std::floor(std::sqrt(double(p)))),
                                                       nodesize(1),
                                                       maxnodes(-1),
                                                       root(NULL) {
}


DecisionTree::DecisionTree(double* train_data,
                           int* train_y,
                           int n,
                           int p,
                           int mtry,
                           int nodesize,
                           int maxnodes) :
                                   train_data(train_data),
                                   train_y(train_y),
                                   n(n),
                                   p(p),
                                   mtry(mtry),
                                   nodesize(nodesize),
                                   maxnodes(maxnodes),
                                   root(NULL) {
}


DecisionTree::~DecisionTree() {
    deleteTree(root);
    return;
}

void DecisionTree::deleteTree(node* t) {
    if (t == NULL) {
        return;
    }

    if(t->left != NULL) {
        deleteTree(t->left);
    }

    if(t->right != NULL) {
        deleteTree(t->right);
    }

    delete t;
    return;
}

void DecisionTree::train() {
    double* device_data;
    int* device_labels;

    data_to_device(&device_data, train_data, n * p);
    data_to_device(&device_labels, train_y, n);

    // Get a random sample of the data.
    double* sample_data;
    int* sample_labels;
    bootstrap_sample(device_data,
                     device_labels,
                     &sample_data,
                     &sample_labels,
                     n,
                     p);

    double** attribute_value_list = new double*[p];
    int** class_label_list = new int*[p];
    int** rid_list = new int*[p];

    for (int i = 0; i < p; i++) {
        init_attribute_list_memory(&attribute_value_list[i],
                                   &class_label_list[i],
                                   &rid_list[i],
                                   n);
    }

    // Build attribute lists.
    /*
    build_attribute_lists(sample_data,
                          sample_labels,
                          n,
                          p,
                          attribute_value_list,
                          class_label_list,
                          rid_list);

    // TODO change back to sample after debugging
*/
    build_attribute_lists(device_data,
                          device_labels,
                          n,
                          p,
                          attribute_value_list,
                          class_label_list,
                          rid_list);


    int best_attr_idx;
    int val_idx;
    double best_gini;

    find_split(attribute_value_list,
               class_label_list,
               rid_list,
               n,
               p,
               &best_attr_idx,
               &val_idx,
               &best_gini);

    std::cout << "best_attr_idx = " << best_attr_idx << std::endl;
    std::cout << "val_idx = " << val_idx << std::endl;
    std::cout << "best_gini = " << best_gini << std::endl;


    // Free all the memory
    for (int i = 0; i < p; i++) {
        free_from_device(attribute_value_list[i]);
        free_from_device(class_label_list[i]);
        free_from_device(rid_list[i]);
    }

    delete[] attribute_value_list;
    delete[] class_label_list;
    delete[] rid_list;

    free_from_device(sample_data);
    free_from_device(sample_labels);
    free_from_device(device_data);
    free_from_device(device_labels);
}

int DecisionTree::count_help(node* t) {
    if (t == NULL) {
        return 0;
    } else {
        return 1 + std::max(count_help(t->left), count_help(t->right));
    }
}

int DecisionTree::count_levels() {
    return count_help(root);
}

