#include <stdio.h>
#include <stdlib.h>

#ifndef DECISIONTREE_H
#define DECISIONTREE_H

struct node {
    int split_var_idx;
    int split_val_idx;

    int size;
    int is_terminal;

    // P(class = 0 | data) and P(class = 1 | data)
    double p_0;
    double p_1;

    // Just use Gini impurity because that is what sklearn and randomForest use
    // by default
    double impurity;

    double** attribute_value_list;
    int** class_label_list;
    int** rid_list;

    node* left;
    node* right;
};

class DecisionTree {
    public:
        DecisionTree(double* train_data, int* train_y, int n, int p);
        DecisionTree(double* train_data,
                     int* train_y,
                     int n,
                     int p,
                     int mtry,
                     int nodesize,
                     int maxnodes);

        ~DecisionTree();
        double eval(double* new_data);
        int count_levels();
        void train();

    private:
        double* train_data;
        int* train_y;
        int n;
        int p;
        int mtry;
        int nodesize;
        int maxnodes;
        node *root;

        double* device_data;
        int* device_labels;


        void grow(node* t);
        void deleteTree(node* t);
        int count_help(node* t);
};

#endif
