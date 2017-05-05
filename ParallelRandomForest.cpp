#include "ParallelRandomForest.h"
#include "csv.h"
#include <vector>
#include <stdlib.h>
#include <string>

#include <iostream>
using namespace std;

#define contains(vec, elem) (find(vec.begin(), vec.end(), elem) != vec.end())

//calling function should alloc and free subset
//requires M < len
void read_from_file(std:: string filename, int **subset, int M) {

  //None of this works because I'm still working out the details of how to not have three copies of the same code for different files
  //comment the function out to make it compile
  if (filename.find("cancer") != std::string::npos) {

    const int col_num = 10;
    int len = 513;
    std::vector<int> random_index;
    for (int i = 0; i < M; i++) {
      subset[i] = new int[col_num];

      //read in random lines into our array
      int read_index;
      do {
	read_index = rand() % len;
      }
      while (contains(random_index, read_index));
      random_index.push_back(read_index);

      //set line

      io::CSVReader<col_num> in(filename);
      in.read_header(io::ignore_extra_column, "thickness","size_uniformity",
		   "shape_uniformity","adhesion","size","nuclei","chromatin",
		   "nucleoli","mitoses","y");


      int *cur = subset[i];
      int line_count = in.get_file_line();
      while (line_count <= read_index) {
	in.read_row(cur[0],cur[1],cur[2],cur[3],cur[4],cur[5],
		    cur[6],cur[7],cur[8],cur[9]);
	line_count++;
      }
    }
  }

  else if (filename.find("loan") != std::string::npos) {

    const int col_num = 17;
    int len = 665159;
    std::vector<int> random_index;
    for (int i = 0; i < M; i++) {
      subset[i] = new int[col_num];

      //read in random lines into our array
      int read_index;
      do {
	read_index = rand() % len;
      }
      while (contains(random_index, read_index));
      random_index.push_back(read_index);

      //set line

      io::CSVReader<col_num> in(filename);
      in.read_header(io::ignore_extra_column, "loan_amnt","funded_amnt","funded_amnt_inv","int_rate","installment","grade","sub_grade","annual_inc","dti","delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","initial_list_status","y");


      int *cur = subset[i];
      int line_count = in.get_file_line();
      while (line_count <= read_index) {
	in.read_row(cur[0],cur[1],cur[2],cur[3],cur[4],cur[5],
		    cur[6],cur[7],cur[8],cur[9],cur[10],cur[11],
		    cur[12],cur[13],cur[14],cur[15],cur[16]);
	line_count++;
      }
    }
  }
  else {
    return; //FIX THIS SO THAT IT HANDLES THE Marketing dataset
  }
}


void build_trees(std::string data_filename, forest *data_forest,
		 int thread_count) {
  int **subset = new int*[4];
  read_from_file("data/cancer/cancer_train.csv", subset, 4);
}

void bag(forest data_forest, solution *sol, int thread_count) {
}

void test(forest data_forest, std::string testing_filename, solution *sol) {
}
