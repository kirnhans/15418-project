#include <stdio.h>
#include <stdlib.h>

#ifndef DECISIONTREE_H
#define DECISIONTREE_H

struct node {
    int split_var;
    double split_val;

    int size;
    int is_terminal;

    // P(class = 0 | data) and P(class = 1 | data)
    double p_0;
    double p_1;

    // Just use Gini impurity because that is what sklearn and randomForest use
    // by default
    double impurity;

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

    private:
        double* train_data;
        int* train_y;
        int n;
        int p;
        int mtry;
        int nodesize;
        int maxnodes;
        node *root;

        double* device_train_data;
        int* device_train_y;
        int* device_data_idx;

        void train();
        void grow(node* t);
        void deleteTree(node* t);
        int* bootstrap();
        int* findSplit(int* data_weight, int* var_mask);
        int* split(int* data_weight, int var, double val);
};

#endif
