import numpy as np
from pandas import DataFrame
from KMeansPlusPlus import KMeansPlusPlus
from KMeans import KMeans
import time
np.random.seed(1234)  # For reproducibility

numElement = 3000
# We create a data set with three sets of 500 points each chosen from a normal distrubution with a standard deviation of 10.
# The means for the distributions from which we sample are:
# (25,45), (-30,5), and (5,-20)
data = DataFrame({'x': 10 * np.random.randn(numElement) + 10, 'y':
                 10 * np.random.randn(numElement) + 15}, columns=list('xy'))
data = data.append(DataFrame(
    {'x': 10 * np.random.randn(numElement) - 10, 'y': 10 * np.random.randn(numElement) -15}, columns=list('xy')))
data = data.append(DataFrame(
    {'x': 10 * np.random.randn(numElement) - 15, 'y': 10 * np.random.randn(numElement) + 10}, columns=list('xy')))

data = data.append(DataFrame(
    {'x': 10 * np.random.randn(numElement) + 5, 'y': 10 * np.random.randn(numElement) - 20}, columns=list('xy')
))
# Grab a scatterplot
import matplotlib.pyplot as plt
plt.scatter(data['x'], data['y'], s=10)
plt.savefig("clusters_scatterplot.png")

file = open("result.txt","a")
# Cluster
file.writelines("\nTotal point: " + str(numElement))
start_time = time.time()
# your code
km = KMeans(data.copy(), 4)
km.cluster()
elapsed_time = time.time() - start_time
file.writelines("\nKmeans: " + str(elapsed_time) + "s" + "s with " + str(km.get_iterations()) + " times")

start_time = time.time()
# your code
kmpp = KMeansPlusPlus(data.copy(), 4)
kmpp.cluster()
elapsed_time = time.time() - start_time
file.writelines("\nKmeans plus plus: " + str(elapsed_time) + "s with " + str(kmpp.get_iterations()) + " times")


# Get a scatterplot that's color-coded by cluster
colors = [
    "red" if x == 0 else "blue" if x == 1 else "green" if x==2 else "yellow" for x in km.clusters]
plt.scatter(data['x'], data['y'], s=5, c=colors)
plt.savefig("k-means-clusters.png")

colors = [
    "red" if x == 0 else "blue" if x == 1 else "green" if x==2 else "yellow" for x in kmpp.clusters]
plt.scatter(data['x'], data['y'], s=5, c=colors)
plt.savefig("k-means-pp-clusters.png")