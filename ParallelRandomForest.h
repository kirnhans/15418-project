#ifndef __PRF_H__
#define __PRF_H__

#include "DecisionTreeRepr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

void build_trees(std::string data_filename, forest *data_forest, int thread_count);

void bag(forest data_forest, solution *sol, int thread_count);

void test(forest data_forest, std::string testing_filename, solution *sol);

#endif
