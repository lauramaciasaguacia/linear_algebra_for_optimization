import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"In this file we create the elbow graphs which are presented in the pdf file. This is done to help in choosing the"
"number of clusters to use. This code takes around 10 minutes to run because it takes 10 samples for each possible N."


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


def normalize_z_score(array):  # We implemented but decided not to use z-score (see pdf.)
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, i])

    return array


df = pd.read_csv('../EastWestAirlinesCluster.csv')  # Load the data

array = df.to_numpy(dtype=np.float32)  # Convert to numpy array

array = normalize_min_max(array)  # We decide to use min max normalization

array = np.delete(array, 0, 1)  # We remove the IDs column (see pdf.)

n_samples = 10

sum_of_variance = []
min_of_variance = []

n_cent_arr = np.arange(2, 15)

for n_centroids in n_cent_arr:  # We test various numbers to clusters
    cen_dist_list = []
    cen_var_list = []
    for i in range(n_samples):
        n_datapoints = array.shape[0]
        centroids = K_means_plus_plus(n_centroids, n_datapoints, array)  # Initialize centroids

        centroid_id = np.zeros(array.shape[0])  # Each datapoint is assigned to a centroid in the following loop

        old_sum = np.inf
        improvement = 1

        while improvement > 0:  # We keep updating the clusters until we reach an equilibrium
            dis_arr = np.zeros(array.shape[0])
            for i in range(array.shape[0]):
                x = array[i]
                centroid_distances = np.zeros(n_centroids)
                for j in range(n_centroids):
                    centroid_distances[j] = np.sum((x - centroids[j]) ** 2)

                centroid_id[i] = np.argmin(centroid_distances)
                dis_arr[i] = np.min(centroid_distances)

            for k in range(n_centroids):
                n_elements = np.count_nonzero(centroid_id == k)
                if n_elements != 0:
                    centroids[k] = np.sum(array, axis=0, where=(centroid_id[:, None] == k)) / n_elements

            dist_sum = sum(dis_arr)
            improvement = old_sum - dist_sum
            old_sum = dist_sum

        var = 0  # We store the sum of variances in this variable
        k = 0
        for k in range(n_centroids):
            n_elements = np.count_nonzero(centroid_id == k)
            if n_elements != 0:
                var += np.sum((array - centroids[k]) ** 2, where=(centroid_id[:, None] == k)) / n_elements

        cen_dist_list.append(dist_sum)  # Note down the sum of the distances within a cluster
        cen_var_list.append(var / n_centroids)  # Note down the average variance

    min_of_variance.append(min(cen_var_list))
    sum_of_variance.append(sum(cen_var_list))

plt.plot(n_cent_arr, sum_of_variance)  # We plot the sum of variances within for each number of clusters.
plt.xlabel("N Clusters")
plt.ylabel("Sum of variances of each cluster")
plt.title("Elbow method for selecting number of clusters")
plt.show()

plt.plot(n_cent_arr, min_of_variance)  # We plot the variance of the best cluster (out of the 10 samples) for each N.
plt.xlabel("N Clusters")
plt.ylabel("Min of variances of each cluster")
plt.title("Elbow method for selecting number of clusters")
plt.show()
