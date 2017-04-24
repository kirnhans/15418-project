import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn
import time

train_fn = "../data/marketing/bank_train.csv"
test_fn = "../data/marketing/bank_test.csv"

train_df = pd.read_csv(train_fn)
test_df = pd.read_csv(test_fn)

# Code categorical variables
for p in ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome", "month"]:
	label_encoder = sklearn.preprocessing.LabelEncoder()
	label_encoder.fit(train_df[p].unique())
	train_df[p] = label_encoder.transform(train_df[p])
	test_df[p] = label_encoder.transform(test_df[p])

cols = train_df.columns.values 

predictors = [p for p in cols if p != "y"]
target = "y"

# Parameters for random forest
# TODO make these optional command line arguments
n = 500
split_n = 85
leaf_n = 75
seed = 1995

# Initalize random forest class
regr = RandomForestClassifier(random_state = seed, n_estimators = n, min_samples_split = split_n, min_samples_leaf = leaf_n)

# Train
train_start = time.time()
regr.fit(train_df[predictors], train_df[target])
train_end = time.time()

# Training time in seconds
train_time = train_end - train_start

# Predict on test data
test_start = time.time()
test_probs = regr.predict_proba(test_df[predictors])
test_end = time.time()

# Testing/predicting time in seconds
test_time = test_end - test_start

# Get predicted probability that y = 1
test_predictions = [x[1] for x in test_probs]

# Calculate MSE for test data
test_error = (test_predictions - test_df[target]) ** 2
mse = test_error.mean()

# Calculate misclassification rate on test data
mis_rate = 1 - accuracy_score(test_df[target], regr.predict(test_df[predictors]))

# Print out summary
print "**************sklearn random forest**************"
print "Marketing data:"
print "Test MSE: " + str(mse)
print "Test misclassification rate: " + str(mis_rate)
print "Train time: " + str(train_time)
print "Predict time: " + str(test_time)

