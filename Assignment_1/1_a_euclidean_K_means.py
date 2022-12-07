import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

array = df.to_numpy(dtype=np.float32)

def normalize_min_max(array):
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

    return array

def normalize_z_score(array): # We decided not to use z-score
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, i])

    return array


array = normalize_min_max(array)

array = np.delete(array, 0, 1)

dist_list = []

n_samples = 10

sum_of_variance = []
min_of_variance = []

n_cent_arr = np.arange(2, 15)

for n_centroids in n_cent_arr:
    # print(n_centroids)
    cen_dist_list = []
    cen_var_list = []
    for i in range(n_samples):
        n_datapoints = array.shape[0]
        centroids = K_means_plus_plus(n_centroids, n_datapoints, array)

        centroid_id = np.zeros(array.shape[0])

        old_sum = np.inf

        improvement = 1
        while improvement > 0:
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

        var = 0
        k = 0
        for k in range(n_centroids):
            n_elements = np.count_nonzero(centroid_id == k)
            if n_elements != 0:
                var += np.sum((array - centroids[k]) ** 2, where=(centroid_id[:, None] == k)) / n_elements
        cen_dist_list.append(dist_sum)
        cen_var_list.append(var / n_centroids)

    min_of_variance.append(min(cen_var_list))
    sum_of_variance.append(sum(cen_var_list))
    dist_list.append(sum(cen_dist_list))

plt.plot(n_cent_arr, sum_of_variance)
plt.xlabel("N Clusters")
plt.ylabel("Sum of variances of each cluster")
plt.title("Elbow method for selecting number of clusters")
plt.show()


plt.plot(n_cent_arr, min_of_variance)
plt.xlabel("N Clusters")
plt.ylabel("Min of variances of each cluster")
plt.title("Elbow method for selecting number of clusters")
plt.show()

