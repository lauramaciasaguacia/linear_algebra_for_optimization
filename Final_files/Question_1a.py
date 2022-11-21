import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"After choosing the number of clusters using the elbow method this code is used to investigate the clusters found with"
"the optimal N. We cluster the data for 10 samples (there is some randomness due to the K-means++ method) and choose"
"the best cluster out of these 10 samples."

n_centroids = 3


def K_means_plus_plus(n_centroids, n_datapoints, array):  # The k-means++ algorithm was used to select initial centroids
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


def normalize_min_max(array):  # min-max normalization
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

    return array


df = pd.read_csv('../EastWestAirlinesCluster.csv')

cols = df.columns

array = df.to_numpy(dtype=np.float32)  # Convert to numpy array

array_copy = np.delete(array, 0, 1)

array = normalize_min_max(array)  # We decide to use min max normalization

array = np.delete(array, 0, 1)  # We remove the IDs column (see pdf.)

n_samples = 10

cent_id_list = []
var_list = []
cent_list = []

for s in range(n_samples):  # We will run for n_samples and use choose the clusters with the best performance
    n_datapoints = array.shape[0]

    centroids = K_means_plus_plus(n_centroids, n_datapoints, array)  # Initialize centroids

    centroid_id = np.zeros(array.shape[0])  # Each datapoint is assigned to a centroid in the following loop

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

    var = 0  # We store the sum of variances in this variable
    k = 0
    for k in range(n_centroids):
        n_elements = np.count_nonzero(centroid_id == k)
        if n_elements != 0:
            var += np.sum((array - centroids[k]) ** 2, where=(centroid_id[:, None] == k)) / n_elements

    var_list.append(var)
    cent_id_list.append(centroid_id)
    cent_list.append(centroids)


best_sample = np.argmin(var_list)
print(best_sample)

best_clustering = cent_id_list[best_sample]
centroids = cent_list[best_sample]

unique, counts = np.unique(best_clustering, return_counts=True)
print(np.asarray((unique.astype(int), np.rint(counts).astype(int))).T)  # Cluster ID vs number of datapoints in cluster

for k in range(n_centroids):  # Convert the centroids from normalized to centroids in the original data
    for i in range(array.shape[1]):
        centroids[k][i] = centroids[k][i] * (np.max(array_copy[:, i]) - np.min(array_copy[:, i])) + np.min(array_copy[:, i])


centroid_df = df = pd.DataFrame()

for i in range(array.shape[1]):
    centroid_df[cols[i+1]] = centroids[:, i]

pd.set_option('display.max_columns', 500)
print(centroid_df)  # Centroid locations in the non-normalized data
