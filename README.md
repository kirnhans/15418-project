# Parallel Random Forest

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

Someone did a [similar project](http://www.news.cs.nyu.edu/~jinyang/pub/biglearning13-forest.pdf)

[Random forest paper (Leo Breiman)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
[Implementation of Random Forest in C++](http://mtv.ece.ucsb.edu/benlee/librf.html)

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

