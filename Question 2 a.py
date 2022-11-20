# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:31:16 2022

@author: Wim
"""

import numpy as np
import pandas as pd
import scipy

# In this file we perform the steps required to solve question 2a. We use the function scipy.sparse.linalg.eigsh to
# compute the N smallest eigenvectors

def normalize(X): #Normalize the data
    for i in range(X.shape[1]):
        val_range = np.max(X[:, i]) - np.min(X[:, i])
        if val_range != 0:
            X[:, i] = (X[:, i] - np.min(X[:, i])) / val_range
    return X

pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array = np.delete(array, 0, 1)
array= normalize(array)

NumberOfClusters = 3  # Decide number of clusters

def kernel_matrix(X):
    li1 = []
    li2 = []
    for i in range(X.shape[1]):
        li1.append(np.subtract(np.repeat([X[:, i]], X.shape[0], axis=0), X[:, i][:, None]) ** 2)

    for i in range(X.shape[1]):
        x = np.add(np.repeat([X[:, i]], X.shape[0], axis=0), X[:, i][:, None])
        x = x + (x == 0)
        li2.append(x)

    chi_top = np.array(li1)
    chi_bottom = np.array(li2)
    K = np.exp(- np.sum((chi_top / chi_bottom), axis=0) )
    return K

# def kernel_matrix(X, g):
#     X_norm = np.sum(X ** 2, axis=-1)
#     K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X,X.T)))  # This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
#     return K

# av = []
# for i in range(array.shape[1]):
#     av.append(np.average(array[:, i]))
# gamma = 0
# for f in range(array.shape[0]):
#     gamma = gamma + np.linalg.norm(array[f, :] - av) ** 2  # No clue how to choose this
# gamma = array.shape[0] / gamma

ker = np.matrix(kernel_matrix(array))

### Build a Laplacian from the Kernel Matrix  ###

def make_similarity(A, sim):
    S= A>=sim
    return S

def make_diagonal(A):
    B = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        B[i, i] = np.sum(A[i,:])
    return B

# similar = make_similarity(ker, 0.5)
# diagonal = make_diagonal(similar)
# lap = diagonal - similar

# print("similarity matrix")
# print(similar)
# print("diagonal matrix")
# print(diagonal)
# print("Laplacian matrix")
# print(lap)

# vecs = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(lap), k=NumberOfClusters, which = 'SM')[1]
# vecs = normalize(vecs)

def K_means_plus_plus(n_centroids, n_datapoints, array): # We now apply the K-means++ algorithm to find inital
    # centroid locations
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


# centroids = K_means_plus_plus(NumberOfClusters, vecs.shape[0], vecs)
# print(centroids)


def K_means_clustering(NumberOfClusters, array, centroids):
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


# clusterlist=[]
# for j in range(NumberOfClusters):
#     clusterlist.append([])
    

# cluster_allocation = K_means_clustering(NumberOfClusters, vecs, centroids)
# Each row (datapoint) is allocated to a cluster from 0 to (N-1) the index within the list
# can be used to access the original datapoint in the original array

#for f in range(len(cluster_allocation)):
#    clusterlist[cluster_allocation[f]].append(f)

def ESD_calculator(clusterlist,kernelmatrix,NumberOfClusters):
    ESD=0
    ClusterSum = np.zeros(NumberOfClusters)

    for k in range(NumberOfClusters):  # This is the sum of each cluster, or third term in the formula
        ClusterSum[k] = (ker[:, clusterlist[k]])[clusterlist[k], :].sum()

    for k in range(NumberOfClusters):
        card = len(clusterlist[k])
        c=ClusterSum[k]
        for f in clusterlist[k]:
            a = kernelmatrix[f, f]
            b = kernelmatrix[f, clusterlist[k]].sum()
            dfk= a-2/card * b + c/(card**2)
            ESD=ESD + dfk
    return ESD

np.set_printoptions(threshold=4000)



# similarity = [0.05,0.25, 0.5, 0.75, 0.95]
# totESD = []
# minESS = 10**8

# for s in similarity:
#     print(s)
#     similar = make_similarity(ker, s)
#     diagonal = make_diagonal(similar)
#     lap = diagonal - similar
#     vecs = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(lap), k=NumberOfClusters, which = 'SM')[1]
#     vecs = normalize(vecs)
#     for i in range(3):
#         print(i)
#         centroids = K_means_plus_plus(NumberOfClusters, vecs.shape[0], vecs)
#         cluster_allocation = K_means_clustering(NumberOfClusters, vecs, centroids)
        
#         clusterlist=[]
#         for j in range(NumberOfClusters):
#             clusterlist.append([])
            
#         for f in range(len(cluster_allocation)):
#             clusterlist[int(cluster_allocation[f])].append(f)
            
#         ESS = 0
#         for i in range(len(clusterlist)):
#             ESS=ESS+ np.var(array[clusterlist[i], :])
#         if ESS<minESS:
#             minESS = ESS
#             print("here")
#             beslist = clusterlist.copy()
#             bests=s
        
# for i in range(len(beslist)):
#     print(len(beslist[i]))

minESS = 0.19681680388748646
similar = ker
diagonal = make_diagonal(similar)
lap = diagonal - similar
vecs = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(lap), k=NumberOfClusters, which = 'SM')[1]
vecs = normalize(vecs)
for i in range(3):
    print(i)
    centroids = K_means_plus_plus(NumberOfClusters, vecs.shape[0], vecs)
    cluster_allocation = K_means_clustering(NumberOfClusters, vecs, centroids)
    
    clusterlist=[]
    for j in range(NumberOfClusters):
        clusterlist.append([])
        
    for f in range(len(cluster_allocation)):
        clusterlist[int(cluster_allocation[f])].append(f)
        
    ESS = 0
    for i in range(len(clusterlist)):
        ESS=ESS+ np.var(array[clusterlist[i], :])
        
    if ESS<minESS:
        print("here")
        beslist2 = clusterlist.copy()
        