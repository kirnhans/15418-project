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

        io::CSVReader<14> in(filename);
	in.read_header(io::ignore_extra_column, "loan_amnt","funded_amnt","funded_amnt_inv","int_rate","installment","annual_inc","dti","delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","y");
        double v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13;
        int y;
        while (in.read_row(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
			   v12, v13, y)) {
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

            labels.push_back(y);
        }

        n = filename.compare("data/loan/loan_train.csv") ? 221721 : 665158;
        p = 13;
    }

    else if(!filename.compare("data/marketing/bank_train.csv") ||
            !filename.compare("data/marketing/bank_test.csv")) {

        io::CSVReader<7> in(filename);
        in.read_header(io::ignore_extra_column, "age", "balance", "duration",
		       "campaign", "pdays", "previous", "y");
        double v1, v2, v3, v4, v5, v6;
        int y;
        while (in.read_row(v1, v2, v3, v4, v5, v6, y)) {
            data.push_back(v1);
            data.push_back(v2);
            data.push_back(v3);
            data.push_back(v4);
            data.push_back(v5);
            data.push_back(v6);

            labels.push_back(y);
        }

        n = filename.compare("data/marketing/bank_train.csv") ?  11304 : 33909 ;
        p = 6;
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
