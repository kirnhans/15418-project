#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#ifndef DATAREADER_H
#define DATAREADER_H

#endif

class DataReader {
    public:
        DataReader() {}
        void read(std::string filename);
        int get_n();
        int get_p();
        double* data_arr();
        int* label_arr();

    private:
        int n;
        int p;
        std::vector<double> data;
        std::vector<int> labels;
};
