import matplotlib.pyplot as plt
import sys
from numpy import genfromtxt

# NumPy function to create array from tabular dataset
my_csv = genfromtxt('../datasets/wine-data.csv', delimiter=',')

# Slicing array
array_flavanoids = my_csv[:, 0]

# Slicing array
array_colorintensity = my_csv[:, 1]

# Scatter plot function
plt.scatter(array_flavanoids, array_colorintensity, c='b', marker='o')
plt.xlabel('Concentration of flavanoids', fontsize=16)
plt.ylabel('Color intensity', fontsize=16)
plt.title('Concentration of flavanoids vs Color intensity', fontsize=20)
plt.show()

sys.exit()