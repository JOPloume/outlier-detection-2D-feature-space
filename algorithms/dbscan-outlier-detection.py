import pandas as pd
import sys
from sklearn.cluster import DBSCAN
from collections import Counter

# Reading in 2D Feature Space
feature_space = pd.read_csv("../datasets/wine-data.csv", header=None, sep=",")
data = feature_space.iloc[:, 0:2].values

# DBSCAN model with parameters
model = DBSCAN(eps=0.8, min_samples=10).fit(data)

# Creating Panda DataFrame with Labels for Outlier Detection
outlier_df = pd.DataFrame(data)

# Printing total number of values for each label
print(Counter(model.labels_))

# Printing DataFrame being considered as Outliers -1
print(outlier_df[model.labels_ == -1])

# Printing and Indicating which type of object outlier_df is
print(type(outlier_df))

# Exporting this DataFrame to CSV
outlier_df[model.labels_ == -1].to_csv("../datasets/dbscan-outliers.csv")

sys.exit()