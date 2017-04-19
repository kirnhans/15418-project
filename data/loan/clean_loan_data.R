library(tidyverse)

# Clear the workspace
rm(list = ls())

# Data from Lending Club (data downloaded from Kaggle)
# https://www.kaggle.com/wendykan/lending-club-loan-data
loan <- read_csv("loan.csv")

# Drop the id column.
loan <- loan[,-1]

cols <-
  c("loan_amnt", 
    "funded_amnt", 
    "funded_amnt_inv", 
    "int_rate", 
    "installment", 
    "grade", 
    "sub_grade", 
    "annual_inc", 
    "dti", 
    "delinq_2yrs", 
    "inq_last_6mths", 
    "open_acc", 
    "pub_rec", 
    "revol_util", 
    "total_acc", 
    "initial_list_status",
    "loan_status")

# Keep specified columns and drop the others.
loan <- loan[, cols]

# Drop missing data.
loan <- loan[complete.cases(loan),]

# Add column that we are going to predict:
# if loan is "bad" e.g. default
loan$bad <- as.factor(ifelse(loan$loan_status == "Fully Paid" |
                               loan$loan_status == "In Grace Period" |
                               loan$loan_status == "Current", 0, 1))

# Write clean data to csv file.
write_csv(loan, "loan_clean.csv")