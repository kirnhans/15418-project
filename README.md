# Parallel Random Forest

## Kirn Hans (khans) and Sally McNichols (smcnicho)

## Summary
We are going to implement random forest on the NVIDIA GPUs in the GHC labs.

In case this proves to be unduly complicated, our alternative is implementing the algorithm using ISPC vectorization.~

## Background
Random forests are an ensemble learning method that is commonly used for classification. Ensemble learning methods use many weak learners together to form an overall prediction that generally has better predictive performance. In the case of random forests, many decision trees are trained from random samples of the data, and after training predictions are made by averaging the results from all of the trees. This process is called bootstrap aggregating (or bagging), and by nature it is parallelizable because each decision tree in the random forest can be trained independently of the others.

## Challenge
While bagging is very parallelizable, the challenge comes from training an individual decision tree. At each node in the decision tree, we need to figure out how to branch. Figuring out how to split can be done in parallel, and splitting nodes at each level in a tree can also be done in parallel, but it is not clear how to balance work well because decision trees are usually not very balanced trees.
We also need to make decisions regarding scheduling of CUDA blocks and the unit of work, such that the granularity is small enough to occupy a large number of blocks and big enough that each CUDA block has sufficient work.

## Resources
We will use the GHC machines.
We will use starter code for Random Forest in C++.

### Existing work
- [Similar project](http://www.news.cs.nyu.edu/~jinyang/pub/biglearning13-forest.pdf)
- [Random forest paper (Leo Breiman)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Implementation of Random Forest in C++](http://mtv.ece.ucsb.edu/benlee/librf.html)
- [sklearn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [R implementation](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)


## Goals and Deliverables
Goal: create a random forest implementation that is competitive with sklearn and R implementations.

Reach goal: scale across multiple GHC machines.

## Platform
C++, CUDA

Alternative: ISPC~

## Schedule
### April 10 - 16
- Project proposal
- Benchmark sklearn and R implementations
- Determine how to divide implementation into smaller tasks
- Implement/Adapt C++ code for Random Forest

### April 17 - 23
- Benchmarking framework for our code
- C++ implementation, parallelize bagging (but not training individual tree)

### April 24~ - 30
- Parallelize node splitting

### May 1 - 7
- Parallelize tree levels

### May 8 - 12
- Final report

~Switch to ISPC by April 24th if we do not have sufficient progress on the CUDA implementation.

# Update [4/25/17]

## Current Progress
We have implemented our testing and benchmarking frameworks. There are 3 main parts to our testing and benchmarking frameworks: random forest implementations in other languages for performance comparison, 3 different datasets for training/testing, and C++ timing code for our CUDA implementation. 

The randomForest package in R and the RandomForestClassifier in sklearn are widely used random forest implementations, so we will be comparing the performance of these implementations to our CUDA implementation to assess the performance of our code. In addition, we have 3 different sized datasets to use for testing (16 KB, 4.6 MB, and 441.8 MB) to help test how well our code scales. We do not have code ready for benchmarking yet, but we do have C++ code prepared for timing our CUDA implementation when we are ready.


## Updated Schedule

* Represent data in C++ **[April 28th (Kirn)]**
	* We need to do this first because we cannot train a decision tree without data.
	* Make sure that library/framework we use can read data from CSV file and allows us to randomly sample rows from the data to use in bootstrap aggregation.

* Represent decision tree in C++ **[April 28th (Sally)]**
	* In order to train decision trees, we need to represent them in some way.
	* Look into how cudaTree, R randomForest, and python sklearn represent decision trees/random forests in code
	* Make sure that this representation will be “CUDA friendly”

* Write sequential code in CUDA kernel that trains a decision tree for a random forest. **[April 30th (Sally)]**
	* Input: data random subset of data
	* Output: decision tree

* Parallelize bagging **[May 2nd (Kirn)]**
	* Use sequential decision tree training code, but train decision trees in parallel.

* Parallelize node splitting **[May 4th (Sally)]**
	* In the training a decision tree code, parallelize the part that decides when and how to split a node.

* Parallelize each level of a tree **[May 7th (Kirn)]**
	* Train a tree so that each tree level is handled in parallel.

* Finish final benchmarking for all code **[May 8th (Kirn)]**

* Visualize performance data **[May 8th (Sally)]**

* Final presentation slideshow **[May 12th (Kirn + Sally)]**

* Final report **[May 12th (Kirn + Sally)]**

## Progress With Respect To Goals

We are behind our original schedule due to overly ambitious goals set for the first two weeks. We did not have specific enough tasks spelled out, so it was difficult to get started on the CUDA implementation of random forest. We were able to finish our performance testing framework, so we at least accomplished that.

Moving forward, we have more clearly figured out the work we have left to do. We have divided this work into manageable tasks now that we have a better idea of implementation details.

We should still be able to achieve our project goals. Not having a specific enough plan is what prevented us from making progress before, but now that we are more organized, we should be able to get a lot of work done in the next three weeks.

## Final Presentation Plans
Looking forward to our final presentation, we plan on creating a slideshow using Google slides. In the slideshow will take some time to explain the basics of random forests, and then explain our project. In addition, we will visualize our performance data by creating plots with ggplot in R. This will require adjusting our benchmarking code to output data into CSV files, but that should be an easy fix.

## Concerning Issues
We are concerned that we are behind schedule with the end of the semester approaching, but we think that we have set ourselves up for success by becoming more organized and making a more specific schedule. 


# Final Report [5/15/17]

## Summary
Our project was parallelizing the machine learning algorithm Random Forest, scaled down to parallelizing its core component Decision Trees. We used CUDA architecture and ran the program on the GHC cluster machines. Our deliverable is parallel code for training decision trees written in CUDA and a comparison against Random Forest code written in Python and Sklearn.

## Background
Decision trees are a common classification method in statistics/machine learning. It takes as an input a training set and a testing set. A decision tree divides the training dataset based on the values of the attribute, e.g. age in the figure shown. Each node is divided into children based on a attribute’s value, e.g. age’s value of 9.5. The value is chosen to divide the dataset to maximize the separation of different classifications, e.g the separation of died and survived. This is called splitting a node.

![Decision Tree]
(https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)

When we are training a tree, we have to determine the attributes and values on which to split nodes. Any attribute can be chosen and any potential value for the attribute can be chosen.
To split a node, we calculate the Gini impurity from the resulting dataset, a measure that indicates the degree of variation. We want to minimize the Gini impurity. The options to calculate over are for each attribute, each value of the attribute, and all the datapoints in the training set for the attribute.

After splitting the node, we split its children and repeat the process until we have very low impurity. The calculations are independent for each subtree. 
Calculating impurity is also independent for each attribute for a given node. Each level however, is completely dependent on the previous level. This calculation is data parallel and probably amenable to SIMD execution. There is little locality because of the independence of the subset for each tree and the randomness of the rows of a single subset.

## Approach


## Results


## References
1. CUDT: https://www.hindawi.com/journals/tswj/2014/745640/
1. SPRINT: http://www.vldb.org/conf/1996/P544.PDF	
1. CSV parsing: https://github.com/ben-strasser/fast-cpp-csv-parser
1. CUDA: 418 Homework 2 starter code
1. Images:
 1. https://en.wikipedia.org/wiki/Decision_tree_learning
 1. https://www.hindawi.com/journals/tswj/2014/745640/

