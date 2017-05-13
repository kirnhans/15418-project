#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "CycleTimer.h"
#include "csv.h"

void train(double* input, int n, int p);
void test(double* input, int n, int p);

int main(int argc, char** argv) {
    std::string train_file("data/cancer/cancer_train.csv");
    std::string test_file("data/cancer/cancer_test.csv");

    // Set RF defaults:
    int ntree = 500;

    // this depends on the dataset, so change later.
    int split_n;

    // The maximumn number of nodes allowed in a tree. If this is -1, then
    // no limit on growing the tree.
    int leaf_n = -1;

    // RNG seed
    int seed = 1995;

    // Training and testing data set sizes
    int n_train = 513;
    int n_test = 172;

    // Number of columns in data.
    int p = 10;

    std::vector<double> train_data;

    io::CSVReader<10> in_train(train_file);
    in_train.read_header(io::ignore_extra_column, "thickness", "size_uniformity",
            "shape_uniformity", "adhesion", "size", "nuclei", "chromatin",
            "nucleoli", "mitoses", "y");
    double v1, v2, v3, v4, v5, v6, v7, v8, v9, v10;
    while (in_train.read_row(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)) {
        train_data.push_back(v1);
        train_data.push_back(v2);
        train_data.push_back(v3);
        train_data.push_back(v4);
        train_data.push_back(v5);
        train_data.push_back(v6);
        train_data.push_back(v7);
        train_data.push_back(v8);
        train_data.push_back(v9);
        train_data.push_back(v10);
    }

    std::vector<double> test_data;

    io::CSVReader<10> in_test(test_file);
    in_test.read_header(io::ignore_extra_column, "thickness", "size_uniformity",
            "shape_uniformity", "adhesion", "size", "nuclei", "chromatin",
            "nucleoli", "mitoses", "y");
    while (in_test.read_row(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)) {
        test_data.push_back(v1);
        test_data.push_back(v2);
        test_data.push_back(v3);
        test_data.push_back(v4);
        test_data.push_back(v5);
        test_data.push_back(v6);
        test_data.push_back(v7);
        test_data.push_back(v8);
        test_data.push_back(v9);
        test_data.push_back(v10);
    }

    // copy training data to device
    double* train_arr = &train_data[0];


    double start_copy_time = CycleTimer::currentSeconds();
    train(train_arr, n_train, p);
    double end_copy_time = CycleTimer::currentSeconds() - start_copy_time;

    std::cout << "copy to device time: " << end_copy_time << " seconds" << std::endl;

    // Time training random forest
    double start = CycleTimer::currentSeconds();

    //build_trees(train_data);

    double train_time = CycleTimer::currentSeconds() - start;

    start = CycleTimer::currentSeconds();
    //test(data_forest, testing_filename);
    double test_time = CycleTimer::currentSeconds() - start;

    printf("----------------------------------------------------------\n");
    std::cout << "Timing Summary for " << train_file <<std::endl;
    printf("----------------------------------------------------------\n");
    std::cout << "Train time: " << train_time << " seconds" << std::endl;
    printf("----------------------------------------------------------\n");
    std::cout << "Test time: " << test_time << " seconds" << std::endl;
    printf("----------------------------------------------------------\n");

    return 0;
}
