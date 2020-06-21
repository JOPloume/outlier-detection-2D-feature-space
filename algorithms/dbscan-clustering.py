import pandas as pd
import matplotlib.pyplot as plt
import sys
from numpy import genfromtxt
from sklearn.cluster import DBSCAN

# Reading in 2D Feature Space
feature_space = pd.read_csv("../datasets/wine-data.csv", header=None, sep=",")
data = feature_space.iloc[:, 0:2].values

# DBSCAN model with parameters
model = DBSCAN(eps=0.8, min_samples=10).fit(data)

# PLOTTING
# NumPy function to create array from tabular dataset
my_csv = genfromtxt('../datasets/wine-data.csv', delimiter=',')

# Slicing array
array_flavanoids = my_csv[:, 0]

# Slicing array
array_colorintensity = my_csv[:, 1]

# Scatter plot function
colors = model.labels_
plt.scatter(array_flavanoids, array_colorintensity, c=colors, marker='o')
plt.xlabel('Concentration of flavanoids', fontsize=16)
plt.ylabel('Color intensity', fontsize=16)
plt.title('Concentration of flavanoids vs Color intensity', fontsize=20)
plt.show()

sys.exit()