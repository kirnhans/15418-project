#include <stdio.h>
#include <stdlib.h>

#include "DataReader.h"
#include "csv.h"

void DataReader::read(std::string filename) {
    // TODO: make this less hard-coded
    if(filename.compare("data/cancer/cancer_train.csv") ||
            filename.compare("data/cancer/cancer_test.csv")) {
        io::CSVReader<10> in(filename);
        in.read_header(io::ignore_extra_column, "thickness", "size_uniformity",
                "shape_uniformity", "adhesion", "size", "nuclei", "chromatin",
                "nucleoli", "mitoses", "y");
        double v1, v2, v3, v4, v5, v6, v7, v8, v9;
        int y;
        while (in.read_row(v1, v2, v3, v4, v5, v6, v7, v8, v9, y)) {
            data.push_back(v1);
            data.push_back(v2);
            data.push_back(v3);
            data.push_back(v4);
            data.push_back(v5);
            data.push_back(v6);
            data.push_back(v7);
            data.push_back(v8);
            data.push_back(v9);

            labels.push_back(y);
        }

        n = filename.compare("data/cancer/cancer_train.csv") ? 513 : 172;
        p = 9;
    }
}

int DataReader::get_n() {
    return p;
}

int DataReader::get_p() {
    return n;
}

double* DataReader::data_arr() {
    return &data[0];
}

int* DataReader::label_arr() {
    return &labels[0];
}

