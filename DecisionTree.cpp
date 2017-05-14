#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

#include "DecisionTree.h"

void data_to_device(double** device_input_data, double* input_data, int size);
void data_to_device(int** device_input_data, int* input_data, int size);
void free_from_device(double* device_data);
void free_from_device(int* device_data);
void bootstrap_sample(int** device_nums);
void init_data_mask(int** data_mask, int n, int val);
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
                int* split_right_count);

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

    free_from_device(t->data_mask);

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
    //std::cout << "Start training" << std::endl;

    //std::cout << "Copy data to device" << std::endl;
    data_to_device(&device_train_data, train_data, n * p);
    data_to_device(&device_train_y, train_y, n);

    //std::cout << "Random sample the data" << std::endl;
    // Find the indices of data we should train on.
    bootstrap_sample(&device_data_idx);

    root = new node();
    root->size = n;
    root->is_terminal = n > 1 ? 0 : 1;
    init_data_mask(&root->data_mask, n, 1);

    //std::cout << "grow tree" << std::endl;
    grow(root);
}

void DecisionTree::grow(node* t) {
    if (t->is_terminal) {
        return;
    }

    node* left = new node();
    node* right = new node();

    find_split(t->data_mask,
               device_data_idx,
               device_train_data,
               device_train_y,
               n,
               p,
               &left->data_mask,
               &right->data_mask,
               &t->split_var,
               &t->split_val,
               &t->impurity,
               &left->size,
               &right->size);


    t->left = left;
    t->right = right;

    if (left->size <= nodesize || t->impurity < 0.0000001) {
        left->is_terminal = 1;
    }

    if (right->size <= nodesize || t->impurity < 0.0000001) {
        right->is_terminal = 1;
    }

    grow(left);
    grow(right);
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

