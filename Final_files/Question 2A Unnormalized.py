import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
# In this file we perform the steps required to solve question 2a with unnormalised clustering.
# We use the function scipy.sparse.linalg.eigsh to compute the N smallest eigenvectors

np.random.seed(0)  # Use of a random seed to make the code reproducable


def normalize_min_max(array):  # min-max normalization
    for i in range( array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

    return array


pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array = np.delete(array, 0, 1)
array2 = array.copy()
array = normalize_min_max(array)
head = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans',
            'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award','Size']
NumberOfClusters = 3  # Decide number of clusters


def kernel_matrix(X, g):
    X_norm = np.sum(X ** 2, axis=-1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X,
                                                                    X.T)))  # This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
    return K


gamma = 2.71428571  # The value obtained in exercise 1

ker = np.matrix(kernel_matrix(array, gamma))  # Build the kernel matrix


### Functions to build a Laplacian from the Kernel Matrix  ###

def make_similarity(A, sim):  # Input of a matrix and a similarity threshold
    S = A >= sim  # S is a matrix where every entry is a boolean whether it is larger than the threshold
    return S


def make_diagonal(A):  # Input of a matrix
    B = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        B[i, i] = np.sum(A[i, :])  # The entries on the diagonal are the sums defined on p132
    return B


### Functions to cluster the data, same as in exercise 1 ###

def K_means_plus_plus(n_centroids, n_datapoints, array):  # We now apply the K-means++ algorithm
    # from exercise 1 to find inital centroid locations
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


def K_means_clustering(NumberOfClusters, array, centroids):  # We now apply the K_means_clustering algorithm
    # from exercise 1 to find the clusters
    centroid_id = np.zeros(array.shape[0])

    old_sum = np.inf

    improvement = 1
    while improvement > 0:
        dist_sum = 0
        for i in range(array.shape[0]):
            x = array[i]
            centroid_distances = np.zeros(NumberOfClusters)
            for j in range(NumberOfClusters):
                centroid_distances[j] = np.sum((x - centroids[j]) ** 2)

            centroid_id[i] = np.argmin(centroid_distances)
            dist_sum += np.min(centroid_distances)

        for k in range(NumberOfClusters):
            n_elements = np.count_nonzero(centroid_id == k)
            if n_elements != 0:
                centroids[k] = np.sum(array, axis=0, where=(centroid_id[:, None] == k)) / n_elements

        improvement = old_sum - dist_sum
        old_sum = dist_sum

    return centroid_id

###Functions to plot the found clusters ###

def Boxplots(array,ClusterList,NumberOfClusters,head): 
    for i in range(array.shape[1]):
        d = []
        for j in range(NumberOfClusters):
            d.append(array[ClusterList[j],i])
        sns.violinplot(data=d)
        plt.title(f'Kernel Clusters on Feature: {head[i]}' )
        plt.xlabel(head[i])
        plt.savefig(f'2AUN{i}.png')  #saves them in the folder
        plt.close()
    return    #returns a boxplot for every cluster

def scatter(array,ClusterList,NumberOfClusters,head): #We are looking if there are any 2-dim dependences between the data
    color=np.zeros(array.shape[0])
    for f in range(array.shape[0]):
        for k in range(NumberOfClusters):
            if f in ClusterList[k]:
                color[f]=k+1
    for i in range(array.shape[1]):
        for j in range(i+1,array.shape[1]):
            plt.scatter(array[:,i], array[:,j], c=color )
            plt.title(f"{head[i]} vs {head[j]}")
            plt.xlabel(head[i])
            plt.ylabel(head[j])
            plt.savefig(f'2AUN{i}vs{j}.png')  # saves them in the folder
            plt.close()
    return  #returns a scatterplot with one parameter on each of the 2 axes
            #different colors represent different clusters




### Comparison of different clusters  ###
# Since the clustering is randomly initialised, we run multiple the clustering
# multiple times and choose the one with the lowest ESS

minESS = 10 ** 8  # Set a very large initial value for the ESS that will evaluate the different clusters


similar = make_similarity(ker, 0.9975)  # Construct the similarity matrix
# We have chosen 0.9975 by playing around with it and choosing the value with the best interpretability
diagonal = make_diagonal(similar)  # Construct the corresponding diagonal matrix
lap = diagonal - similar  # Construct the corresponding Laplacian
vecs = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(lap), k=NumberOfClusters, which='SM')[1]
# Compute the eigenvectors corresponding to the k smallest eigenvalues.
# Unlike for the normalised laplacian it goes fast here



for i in range(10):
    centroids = K_means_plus_plus(NumberOfClusters, vecs.shape[0], vecs)
    cluster_allocation = K_means_clustering(NumberOfClusters, vecs, centroids)

    # The cluster_allocation is a list of 4000 entries where every entry is either 0,1, ..., k-1
    # where this value defines in which cluster the corresponding element is

    # We wanted to have a list for every cluster that contains which elements are in it
    # and that is implemented below: clusterlist is a partition of [3999] in k clusters

    clusterlist = []
    for j in range(NumberOfClusters):
        clusterlist.append([])

    for f in range(len(cluster_allocation)):
        clusterlist[int(cluster_allocation[f])].append(f)

    ESS = 0  # We use the ESS to compare different clusters, as suggested in the assignment
    for j in range(NumberOfClusters):
        n_elements = np.count_nonzero(cluster_allocation == j)
        if n_elements != 0:
            ESS += np.sum((vecs - centroids[j]) ** 2, where=(cluster_allocation[:, None] == j))

    if ESS < minESS:  # Compare the lastly computed ESS with the smallest one computed yet
        minESS = ESS
        beslist = clusterlist.copy()  # Copy the corresponding clusterlist in another list such that we don't lose it

for i in range(len(beslist)):  # To get a feeling for the found clustering,
    print(len(beslist[i]))  # the actual beslist can be obtained in the variable explorer


Boxplots(array2,beslist,NumberOfClusters,head)
scatter(array2,beslist,NumberOfClusters,head)



