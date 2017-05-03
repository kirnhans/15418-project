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
    const int col_num = 1;
    int len = 172;
    io::CSVReader<col_num> in(filename);
    /*in.read_header(io::ignore_extra_column, "thickness","size_uniformity",
		   "shape_uniformity","adhesion","size","nuclei","chromatin",
		   "nucleoli","mitoses","y");
    */
    in.read_header(io::ignore_extra_column, "y");
    std::vector<int> random_index;
    for (int i = 0; i < M; i++) {
      //read in random lines into our array
      int read_index;
      do {
	read_index = rand() % len;
      }
      while (contains(random_index, read_index));
      random_index.push_back(read_index);

      //remove
      read_index=i;


      //set line
      //in.set_file_line(read_index);
      //char *line = in.next_line();
      int *cur = subset[i];
      int a;
      //in.read_row(a,cur[1],cur[2],cur[3],cur[4],cur[5],
      //	   cur[6],cur[7],cur[8],cur[9]);
      printf("a ini %d\n",a);
      in.read_row(a);
      printf("a %d\n", a);
      /* printf("line number %d: ", read_index);
      for (int j = 0; j < col_num; j++) {
	printf("%d, ", subset[i][j]);
      }
      printf("\n");*/
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
