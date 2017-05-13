#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "CycleTimer.h"
# include "csv.h"

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

    // If arguments are specified, set them instead of using defaults
    if (argc == 7) {
        train_file = argv[1];
        test_file = argv[2];
      	ntree = atoi(argv[3]);
      	split_n = atoi(argv[4]);
      	leaf_n = atoi(argv[5]);
      	seed = atoi(argv[6]);
    }


    // Time training random forest
    double start = CycleTimer::currentSeconds();

    //build_trees(dataset_filename, &data_forest, thread_count);

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
