import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
print(df.columns)

array = df.to_numpy(dtype=np.float32)


for i in range(1, array.shape[1]):
    array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

array = np.delete(array, 0, 1)

n_centroids = 5

centroids = np.random.rand(n_centroids, array.shape[1])

centroid_id = np.zeros(array.shape[0])

old_sum = np.inf
dist_sum = 0

improvement = 1
while improvement > 0:
    for i in range(array.shape[0]):
        x = array[i]
        centroid_distances = np.zeros(n_centroids)
        for j in range(n_centroids):
            centroid_distances[j] = np.sum((x - centroids[j]) ** 2)

        centroid_id[i] = np.argmin(centroid_distances)
        dist_sum += np.min(centroid_distances)

    for k in range(n_centroids):
        n_elements = np.count_nonzero(centroid_id == k)
        if n_elements != 0:
            centroids[k] = np.sum(array, axis=0, where=(centroid_id[:, None] == k)) / n_elements

    improvement = old_sum - dist_sum
    old_sum = dist_sum
    dist_sum = 0

print(centroids)