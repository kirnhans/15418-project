library(tidyverse)

# Clear the workspace.
rm(list = ls())

# Read in data.
bank <- read_delim("bank-full.csv", ";")

# Recode outcome to 1 if yes and 0 if no.
bank$y <- ifelse(bank$y == "yes", 1, 0)

# Remove any missing data.
bank_clean <- bank[complete.cases(bank),]

# Write to csv.
write_csv(bank_clean, "bank_clean.csv")

# Split data for training and testing. Use 75/25 train/test split.
set.seed(1995)
smp_size <- floor(0.75 * nrow(bank_clean))

train_ind <- sample(seq_len(nrow(bank_clean)), size = smp_size)

train <- bank_clean[train_ind, ]
test <- bank_clean[-train_ind, ]

write_csv(train, "bank_train.csv")
write_csv(test, "bank_test.csv")