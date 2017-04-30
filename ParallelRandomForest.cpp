#include "ParallelRandomForest.h"
#include "csv.h"
#include <vector>
#include <stdlib.h>
#include <string>

#define contains(vec, elem) (find(vec.begin(), vec.end(), elem) != vec.end())

//calling function should alloc and free subset
//requires M < len
void read_from_file(std:: string filename, int **subset, int M) {
  //None of this works because I'm still working out the details of how to not have three copies of the same code for different files
  //comment the function out to make it compile
  if (filename.find("cancer") != std::string::npos) {
    int col_num = 1;
    int len = 172;
    io::CSVReader<1> in(filename);
    /* in.read_header(io::ignore_extra_column, "thickness","size_uniformity",
		   "shape_uniformity","adhesion","size","nuclei","chromatin",
		   "nucleoli","mitoses","y");
    */
    in.read_header(io::ignore_extra_column, "thickness");
    std::vector<int> random_index;
    for (int i = 0; i < M; i++) {
      //read in random lines into our array
      int read_index;
      do {
	read_index = rand() % len;
      }
      while (contains(random_index, read_index));
      random_index.push_back(read_index);
      //set line
      in.set_file_line(read_index);
      char *line = in.next_line();
      for (int j = 0; j < col_num; j++) {
	subset[i][j] = line[j];
      }
    }
  }

  else {
    return; //FIX THIS SO THAT IT HANDLES THE OTHER DATASETS
  }
}

void build_trees(std::string data_filename, forest *data_forest,
		 int thread_count) {
  int **subset = new int*[4];
  for (int i = 0; i < 4; i++)
    subset[i] = new int[10];
  read_from_file("data/cancer/cancer.csv", subset, 4);
}

void bag(forest data_forest, solution *sol, int thread_count) {
}

void test(forest data_forest, std::string testing_filename, solution *sol) {
}
