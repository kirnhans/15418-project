# Clear the workspace
rm(list = ls())

# Load the data.  NAs are coded as "?"
# Data from: 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
dat <- read.csv('breast-cancer-wisconsin.data', 
               header = FALSE, 
               na.strings = "?")

# Label the data fields.  Description and values are as noted.
names(dat) = c(
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
  'class' # Class:            (2 for benign, 4 for malignant)
)

#Drop ID number, we don't want to use this for anything
dat <- dat[,-1]

#Change outcome to be descriptive
# 0 = benign, 1 = malignant
dat$class <- ifelse(dat$class == 2, 0, 1)

# Make outcome a factor.
dat$class <- as.factor(dat$class)

# Drop missing data.
dat <- dat[complete.cases(dat),]

# Write data to csv file for easy use.
write.csv(dat, file = "cancer.csv", row.names = FALSE)
