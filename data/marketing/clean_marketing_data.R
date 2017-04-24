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