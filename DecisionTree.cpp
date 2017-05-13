#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "DecisionTree.h"

void data_to_device(double** device_input_data, double* input_data, int size);
void data_to_device(int** device_input_data, int* input_data, int size);
void free_from_device(double* device_data);
void free_from_device(int* device_data);
void bootstrap_sample(int** device_nums);

DecisionTree::DecisionTree(double* train_data, int* train_y, int n, int p) :
                                                       train_data(train_data),
                                                       train_y(train_y),
                                                       n(n),
                                                       p(p),
                                                       mtry(std::floor(std::sqrt(double(p)))),
                                                       nodesize(1),
                                                       maxnodes(-1),
                                                       root(NULL) {
    train();
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
    train();
}


DecisionTree::~DecisionTree() {
    free_from_device(device_train_data);
    free_from_device(device_train_y);
    free_from_device(device_data_idx);
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
    // Copy the training data to the device
    data_to_device(&device_train_data, train_data, n * p);
    data_to_device(&device_train_y, train_y, n);

    // Find the indices of data we should train on.
    bootstrap_sample(&device_data_idx);

    root = new node();
    root->size = n;
    root->is_terminal = n > 1 ? 0 : 1;
    grow(root);
}

void DecisionTree::grow(node* t) {
    if (t->is_terminal) {
        return;
    }

    // find split

    // calculate split

    // make children

    // grow children
}
