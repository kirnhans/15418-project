#include <stdio.h>
#include <stdlib.h>

#ifndef DECISIONTREE_H
#define DECISIONTREE_H

struct node {
    int split_var;
    double split_val;
    double impurity;
    int size;
    int is_terminal;
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
        void train();
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

        void deleteTree(node* t);
        int* bootstrap();
        int* findSplit(int* data_weight, int* var_mask);
        int* split(int* data_weight, int var, double val);
};

#endif
