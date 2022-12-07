import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n_centroids = 3

def K_means_plus_plus(n_centroids, n_datapoints, array):
    problist = []
    centroids = []

    for i in range(n_datapoints):
        problist.append(1 / n_datapoints)

    v = np.random.choice(n_datapoints, p=problist)
    centroids.append(array[v])

    for k in range(n_centroids - 1):
        distlist = []  # The last iteration this is not really necessary
        for i in range(n_datapoints):
            dist_options = []
            for j in range(len(centroids)):
                dist_options.append((np.linalg.norm(centroids[j] - array[i])) ** 2)
            dist = min(dist_options)
            distlist.append(dist)

        for i in range(n_datapoints):
            prob = distlist[i] / np.sum(distlist)
            problist[i] = prob

        v = np.random.choice(n_datapoints, p=problist)
        centroids.append(array[v])

    centroids = np.array(centroids)

    return centroids


pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')

cols = df.columns

array = df.to_numpy(dtype=np.float32)

array_copy = np.delete(array, 0, 1)


for i in range(1, array.shape[1]):
    array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

array = np.delete(array, 0, 1)

n_datapoints = array.shape[0]

dist_list = []

centroids = K_means_plus_plus(n_centroids, n_datapoints, array)

centroid_id = np.zeros(array.shape[0])

old_sum = np.inf

improvement = 1
while improvement > 0:
    dist_sum = 0
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

for k in range(n_centroids):
    for i in range(array.shape[1]):
        centroids[k][i] = centroids[k][i] * (np.max(array_copy[:, i]) - np.min(array_copy[:, i])) + np.min(array_copy[:, i])


centroid_df = df = pd.DataFrame()

for i in range(array.shape[1]):
    centroid_df[cols[i+1]] = centroids[:, i]

print(centroid_df)