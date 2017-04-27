#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "CycleTimer.h"
#include "ParallelRandomForest.h"
#include "DecisionTreeRepr.h"

#define USE_BINARY_GRAPH 1


void load_answer(std::string filename, solution *sol) {
  //FILL THIS IN
}


int main(int argc, char** argv) {

    int  num_threads = -1;

    if (argc < 6)
    {
        std::cerr << "Usage: <path/to/graph/file> [manual_set_thread_count]\n";
        std::cerr << "To get results across all thread counts: <path/to/graph/file>\n";
        std::cerr << "Run with certain threads count (no correctness run): <path/to/graph/file> <thread_count>\n";
        exit(1);
    }

    int thread_count = -1;
    int n;
    int split_n;
    int leaf_n;
    int seed;
    forest data_forest;

    if (argc == 9) {
        thread_count = atoi(argv[4]);
      	n = atoi(argv[5]);
      	split_n = atoi(argv[6]);
      	leaf_n = atoi(argv[7]);
      	seed = atoi(argv[8]);
    }

    std::string dataset_filename = argv[1];
    std::string testing_filename = argv[2];
    std::string answer_filename = argv[3];

    // If we want to run on all threads
    if (thread_count <= -1) {
      // Static assignment to get consistent usage across trials
      int max_threads = 500; //arbitrary

      // static num_threadss
      std::vector<int> num_threads;

      // dynamic num_threads
      for (int i = 1; i < max_threads; i *= 2) {
        num_threads.push_back(i);
      }

      num_threads.push_back(max_threads);
      int n_usage = num_threads.size();

      solution sol1;
      sol1.nodes = new sol_node[n];
      solution sol2;
      sol2.nodes = new sol_node[n];

      double base;
      double training_time;

      double ref_base;
      double ref_time;

      double start;
      std::stringstream timing;
      std::stringstream ref_timing;
      std::stringstream relative_timing;

      bool check = true;

      timing << "Threads  Parallel\n";
      ref_timing << "Threads  Reference\n";
      relative_timing << "Threads  Comparison\n";

      //Loop through assignment values;
      for (int i = 0; i < n_usage; i++)
        {
	  printf("----------------------------------------------------------\n");
	  std::cout << "Running with " << num_threads[i] << " threads\n";

	  // Time training random forest
	  start = CycleTimer::currentSeconds();
	  build_trees(dataset_filename, &data_forest, thread_count);
	  training_time = CycleTimer::currentSeconds() - start;

	  start = CycleTimer::currentSeconds();
	  test(data_forest, testing_filename, &sol1);
	  double test_time = CycleTimer::currentSeconds() - start;

	  load_answer(answer_filename, &sol2);
	  std::cout << "Testing Correctness \n";
	  for (int j=0; j<sol1.num_nodes; j++) {
	    if (sol1.nodes[j].prediction != sol2.nodes[j].prediction) {
	      fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j,
		      sol1.nodes[j].prediction, sol2.nodes[j].prediction);
	      check = false;
	      break;
	    }
	  }

	  if (i == 0)
            {
	      base = training_time;
            }

	  char buf[1024];
	  char ref_buf[1024];
	  char relative_buf[1024];

	  sprintf(buf, "%4d:   %.4f (%.4fx)\n",
		  num_threads[i], training_time, base/training_time);

	  //need to read timing data from bash script
	  //FIX
	  /*            sprintf(ref_buf, "%4d:   %.4f (%.4fx)\n",
			num_threads[i], ref_top_time, ref_top_base/ref_top_time);
			sprintf(relative_buf, "%4d:   %.2fp\n",
			num_threads[i], 100*top_time/ref_top_time);
	  */

	  timing << buf;
	  ref_timing << ref_buf;
	  relative_timing << relative_buf;
        }

        printf("----------------------------------------------------------\n");
        std::cout << "Timing Summary" << std::endl;
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Reference Summary" << std::endl;
        std::cout << ref_timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "For grading reference (based on execution times)" << std::endl << std::endl;
        std::cout << "Correctness: " << std::endl;
        if (!check)
            std::cout << "Not Correct" << std::endl;
        std::cout << std::endl << "Timing: " << std::endl <<  relative_timing.str();
    }
    //Run the code with only one thread count and only report speedup
    else
    {
      bool check = true;

      solution sol1;
      sol1.nodes = new sol_node[n];
      solution sol2;
      sol2.nodes = new sol_node[n];

      double base;
      double training_time;

      double ref_base;
      double ref_time;

      double start;
      std::stringstream timing;
      std::stringstream ref_timing;
      std::stringstream relative_timing;


      timing << "Threads  Random_Forest\n";
      //ref_timing << "Threads  Top Down    Bottom Up   Hybrid\n";

        //Loop through assignment values;
      std::cout << "Running with " << thread_count << " threads" << std::endl;


      // Time training random forest
      start = CycleTimer::currentSeconds();
      build_trees(dataset_filename, &data_forest, thread_count);
      training_time = CycleTimer::currentSeconds() - start;

      start = CycleTimer::currentSeconds();
      test(data_forest, testing_filename, &sol1);
      double test_time = CycleTimer::currentSeconds() - start;

      load_answer(answer_filename, &sol2);
      std::cout << "Testing Correctness \n";
      for (int j=0; j<sol1.num_nodes; j++) {
        if (sol1.nodes[j].prediction != sol2.nodes[j].prediction) {
          fprintf(stderr, "*** Results disagree at %d: %d, %d\n", j,
      		sol1.nodes[j].prediction, sol2.nodes[j].prediction);
      	  check = false;
      	  break;
      	}
      }

      char buf[1024];
      char ref_buf[1024];


      sprintf(buf, "%4d:   %.4f (%.4fx)\n",
	      thread_count, training_time, base/training_time);

      //need to read timing data from bash script
      /*            sprintf(ref_buf, "%4d:   %.4f (%.4fx)\n",
                    num_threads[i], ref_top_time, ref_top_base/ref_top_time);
		    sprintf(relative_buf, "%4d:   %.2fp\n",
                    num_threads[i], 100*top_time/ref_top_time);
      */

      timing << buf;
      ref_timing << ref_buf;
      if (!check)
      	std::cout << "Not Correct" << std::endl;
      printf("----------------------------------------------------------\n");
      std::cout << "Timing Summary" << std::endl;
      std::cout << timing.str();
      printf("----------------------------------------------------------\n");
      std::cout << "Reference Summary" << std::endl;
      std::cout << ref_timing.str();
      printf("----------------------------------------------------------\n");
    }

    return 0;
}
