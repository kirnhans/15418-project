# Clear the workspace
rm(list = ls())

# Load the data.  NAs are coded as "?"
# Data from: 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
cancer <- read.csv('breast-cancer-wisconsin.data', 
               header = FALSE, 
               na.strings = "?")

# Label the data fields.  Description and values are as noted.
names(cancer) = c(
  'id', # Sample code number            id number
  'thickness', # Clump Thickness               1 - 10
  'size_uniformity', # Uniformity of Cell Size       1 - 10
  'shape_uniformity', # Uniformity of Cell Shape      1 - 10
  'adhesion', # Marginal Adhesion             1 - 10
  'size', # Single Epithelial Cell Size   1 - 10
  'nuclei', # Bare Nuclei                   1 - 10
  'chromatin', # Bland Chromatin               1 - 10
  'nucleoli', # Normal Nucleoli               1 - 10
  'mitoses', # Mitoses                       1 - 10
  'y' # Class:            (2 for benign, 4 for malignant)
)

#Drop ID number, we don't want to use this for anything
cancer <- cancer[,-1]

# Change outcome to be descriptive
# 0 = benign, 1 = malignant
cancer$y <- ifelse(cancer$y == 2, 0, 1)

# Make outcome a factor.
cancer$y <- as.factor(cancer$y)

# Drop missing data.
cancer <- cancer[complete.cases(cancer),]

# Write data to csv file for easy use.
write.csv(cancer, file = "cancer.csv", row.names = FALSE)

# Split data for training and testing. Use 75/25 train/test split.
set.seed(1995)
smp_size <- floor(0.75 * nrow(cancer))

train_ind <- sample(seq_len(nrow(cancer)), size = smp_size)

train <- cancer[train_ind, ]
test <- cancer[-train_ind, ]

write_csv(train, "cancer_train.csv")
write_csv(test, "cancer_test.csv")