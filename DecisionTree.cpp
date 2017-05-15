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
                          int* left_n);

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
    free_from_device(device_data);
    free_from_device(device_labels);

    for (int i = 0; i < p; i++) {
        free_from_device(root->attribute_value_list[i]);
        free_from_device(root->class_label_list[i]);
        free_from_device(root->rid_list[i]);
    }

    deleteTree(root);
    return;
}

void DecisionTree::deleteTree(node* t) {
    if (t == NULL) {
        return;
    }

    delete[] t->attribute_value_list;
    delete[] t->class_label_list;
    delete[] t->rid_list;

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
    //std::cout << "start training" << std::endl;
    data_to_device(&device_data, train_data, n * p);
    data_to_device(&device_labels, train_y, n);


    double** attribute_value_list = new double*[p];
    int** class_label_list = new int*[p];
    int** rid_list = new int*[p];

    for (int i = 0; i < p; i++) {
        init_attribute_list_memory(&attribute_value_list[i],
                                   &class_label_list[i],
                                   &rid_list[i],
                                   n);
    }

    build_attribute_lists(device_data,
                          device_labels,
                          n,
                          p,
                          attribute_value_list,
                          class_label_list,
                          rid_list);

    root = new node();
    root->size = n;
    root->attribute_value_list = attribute_value_list;
    root->class_label_list = class_label_list;
    root->rid_list = rid_list;

    //std::cout << "grow root" << std::endl;
    grow(root);
}

void DecisionTree::grow(node* t) {
    //std::cout << "start grow" << std::endl;
    if (t->size == 1 || t->is_terminal == 1) {
        t->is_terminal = 1;
        return;
    }

    int best_attr_idx;
    int val_idx;
    double best_gini;

    //std::cout << "find split" << std::endl;

    find_split(t->attribute_value_list,
               t->class_label_list,
               t->rid_list,
               t->size,
               p,
               &best_attr_idx,
               &val_idx,
               &best_gini);

    t->split_var_idx = best_attr_idx;
    t->split_val_idx = val_idx;
    t->impurity = best_gini;

    //std::cout << "best_attr_idx = " << best_attr_idx << std::endl;
    //std::cout << "val_idx = " << val_idx << std::endl;
    //std::cout << "best_gini = " << best_gini << std::endl;

    double** right_attribute_value_list = new double*[p];
    int** right_class_label_list = new int*[p];
    int** right_rid_list = new int*[p];

    double** left_attribute_value_list = new double*[p];
    int** left_class_label_list = new int*[p];
    int** left_rid_list = new int*[p];

    int right_n;
    int left_n;

    //std::cout << "split lists" << std::endl;
    split_attribute_list(t->attribute_value_list,
                         t->class_label_list,
                         t->rid_list,
                         t->size,
                         p,
                         best_attr_idx,
                         val_idx,
                         right_attribute_value_list,
                         left_attribute_value_list,
                         right_class_label_list,
                         left_class_label_list,
                         right_rid_list,
                         left_rid_list,
                         &right_n,
                         &left_n);


    if (right_n > 0) {
        //std::cout << "grow right" << std::endl;
        t->right = new node();

        if (right_n <= 1 || best_gini < 0.04) {
            t->right->is_terminal = 1;
        }

        t->right->attribute_value_list = right_attribute_value_list;
        t->right->class_label_list = right_class_label_list;
        t->right->rid_list = right_rid_list;
        t->right->size = right_n;

        //std::cout << "right size = " << right_n << std::endl;

        grow(t->right);
    }

    if (left_n > 0) {
        //std::cout << "grow left" << std::endl;
        t->left = new node();

        if (left_n <= 1 || best_gini < 0.04) {
            t->left->is_terminal = 1;
        }

        t->left->attribute_value_list = left_attribute_value_list;
        t->left->class_label_list = left_class_label_list;
        t->left->rid_list = left_rid_list;
        t->left->size = left_n;

        //std::cout << "left size = " << left_n << std::endl;

        grow(t->left);
    }
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

