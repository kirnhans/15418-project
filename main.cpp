#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "CycleTimer.h"
#include "csv.h"
#include "DecisionTree.h"
#include "DataReader.h"

int main(int argc, char** argv) {
    // Set RF defaults:
    int ntree = 500;

    // this depends on the dataset, so change later.
    int split_n;

    // The maximumn number of nodes allowed in a tree. If this is -1, then
    // no limit on growing the tree.
    int leaf_n = -1;

    // RNG seed
    int seed = 1995;


    //cancer dataset
     std::string train_file("data/cancer/cancer_train.csv");
    std::string test_file("data/cancer/cancer_test.csv");

    // Training and testing data set sizes
    int n_train = 513;
    int n_test = 172;

    // Number of columns in data.
    int p = 10;

    //loan dataset
    /*std::string train_file("data/loan/loan_train.csv");
    std::string test_file("data/loan/loan_test.csv");

    // Training and testing data set sizes
    int n_train = 665158;
    int n_test = 221721;

    // Number of columns in data.
    int p = 13;

    //marketing dataset
    std::string train_file("data/marketing/bank_train.csv");
    std::string test_file("data/marketing/bank_test.csv");

    // Training and testing data set sizes
    int n_train = 33909;
    int n_test = 11304;

    // Number of columns in data.
    int p = 6;*/

    // Read in the training data
    DataReader *train_reader = new DataReader();
    train_reader->read(train_file);

    // Convert the data into arrays for copying to the GPU.
    double* train_data_arr = train_reader->data_arr();
    int* train_y_arr = train_reader->label_arr();

    std::cout << "first elem: " << train_data_arr[0] << std::endl;
    std::cout << "second elem: " << train_data_arr[1] << std::endl;
    std::cout << "third elem: " << train_data_arr[2] << std::endl;

    double start_train_time = CycleTimer::currentSeconds();

    DecisionTree* tree = new DecisionTree(train_data_arr, train_y_arr, n_train, p - 1);
    tree->train();
    double train_time = CycleTimer::currentSeconds() - start_train_time;

    std::cout << "train time: " << train_time << " seconds" << std::endl;

    std::cout << "decision tree depth: " << tree->count_levels() << std::endl;


    double start = CycleTimer::currentSeconds();
    //test(data_forest, testing_filename);
    double test_time = CycleTimer::currentSeconds() - start;

    printf("----------------------------------------------------------\n");
    std::cout << "Timing Summary for " << train_file <<std::endl;
    printf("----------------------------------------------------------\n");
    std::cout << "Train time: " << train_time << " seconds" << std::endl;
    printf("----------------------------------------------------------\n");
    std::cout << "Test time: " << test_time << " seconds" << std::endl;
    printf("----------------------------------------------------------\n");

    delete tree;

    return 0;
}
