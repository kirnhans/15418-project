library(tidyverse)
library(randomForest)

train_fn <- "../data/loan/loan_train.csv"
test_fn <- "../data/loan/loan_test.csv"

train <- read_csv(train_fn)
test <- read_csv(test_fn)

cat_var <- c("grade", "sub_grade", "initial_list_status")

# Code categorical variables as factors.
for (p in cat_var) {
  train[p] <- factor(unlist(train[p]))
  test[p] <- factor(unlist(test[p]))
}

# Parameters for random forest.
# TODO make these optional command line arguments.
n <- 500
leaf_n <- 75
set.seed(1995)

# Train.
y_idx <- which(colnames(train) == "y")
train_start <- Sys.time()
forest <- randomForest(x = train[,-y_idx], y = factor(train$y),
                       nodesize = leaf_n, ntree = n)
train_end <- Sys.time()

# Training time in seconds.
train_time <- as.double(train_end - train_start)

# Predict probabilities for the test data.
test_start <- Sys.time()
test_prob <- predict(forest, newdata = test[,-y_idx], type = "prob")[,2]
test_end <- Sys.time()

# Testing/predicting time in seconds.
test_time <- as.double(test_end - test_start)

# Predict classes for the test data.
test_pred <- predict(forest, newdata = test[,-y_idx])

# Calculate MSE for test data.
mse <- mean((test_prob - test$y)^2)

# Calculate misclassification rate on test data.
mis_rate <- mean(test_pred != test$y)

# Print out summary
print("**************R randomForest**************")
print("Loan data:")
print(sprintf("Test MSE: %f", mse))
print(sprintf("Test misclassification rate: %f", mis_rate))
print(sprintf("Train Time: %f %s", train_time, 
              units(train_end - train_start)))
print(sprintf("Predict time: %f %s", test_time,
              units(test_end - test_start)))
