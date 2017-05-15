#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "DataReader.h"
#include "csv.h"

void DataReader::read(std::string filename) {
    // TODO: make this less hard-coded
    if(!filename.compare("data/cancer/cancer_train.csv") ||
            !filename.compare("data/cancer/cancer_test.csv")) {

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

        n = filename.compare("data/cancer/cancer_train.csv") ? 172 : 513;
        p = 9;
    }

    else if (!filename.compare("data/loan/loan_train.csv") ||
            !filename.compare("data/loan/loan_test.csv")) {

        io::CSVReader<17> in(filename);
	in.read_header(io::ignore_extra_column, "loan_amnt","funded_amnt","funded_amnt_inv","int_rate","installment","grade","sub_grade","annual_inc","dti","delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","initial_list_status","y");
        double v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
	    v15, v16;
        int y;
        while (in.read_row(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
			   v12, v13, v14, v15, v16, y)) {
            data.push_back(v1);
            data.push_back(v2);
            data.push_back(v3);
            data.push_back(v4);
            data.push_back(v5);
            data.push_back(v6);
            data.push_back(v7);
            data.push_back(v8);
            data.push_back(v9);
            data.push_back(v10);
            data.push_back(v11);
            data.push_back(v12);
            data.push_back(v13);
            data.push_back(v14);
            data.push_back(v15);
            data.push_back(v16);

            labels.push_back(y);
        }

        n = filename.compare("data/loan/loan_train.csv") ?  665158 : 221721;
        p = 16;
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
