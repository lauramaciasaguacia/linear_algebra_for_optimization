# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:01:27 2022

@author: Wim
"""

import numpy as np
import pandas as pd
#import scipy

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
NumberOfClusters=3

def kernel_matrix(X, g):
    X_norm = np.sum(X ** 2, axis=-1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X,X.T)))  # This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
    return K

av = []
for i in range(array.shape[1]):
    av.append(np.average(array[:, i]))
gamma = 0
for f in range(array.shape[0]):
    gamma = gamma + np.linalg.norm(array[f, :] - av) ** 2  # No clue how to choose this
gamma = array.shape[0] / gamma

ker=kernel_matrix(array,gamma)

def make_similarity(A, sim):
    S= A>=sim
    return S

def make_diagonal(A):
    B = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        B[i, i] = np.sum(A[i,:])
    return B

similar = make_similarity(ker, 0.95)
diagonal = make_diagonal(similar)
lap = diagonal - similar
#lap = scipy.sparse.csr_matrix(lap)

def arnoldi(A, num):                         #Implementation of Algorithm 6 p44
    q=np.zeros([A.shape[1],num+1])           #Add 1 to num because we start counting at 0
    b=np.zeros(num+1)
    b[0]=0
    v=np.random.rand(A.shape[0])
    #q[:,0]=np.zeros(len(v))
    q[:,1]= v/np.linalg.norm(v)
    q=np.asmatrix(q)
    for k in range(1,num):
        Aq=np.dot(A,q[:,k])
        a=float(np.dot(q[:,k].T,Aq))
        w= Aq-a*q[:,k]-b[k-1]*q[:,k-1]
        b[k]= np.linalg.norm(w)
        q[:,k+1]=(w/b[k])
        
    Q = np.delete(q, 0, 1)
    return Q


def get_norm(v):
    norm=np.sqrt(np.sum( v**2 ))

    return norm

def Household_Matrix(v,Numb):     #Implementation of householder reflection  matrices
    v=np.array(v)
    m=len(v)
    e1=np.zeros_like(v)
    e1[0]=1
    r=get_norm(v)
    if np.sign(v[0])==0:
        sign=1
    else:
        sign=np.sign(v[0])
    w = v+sign*r*e1
    w = w/get_norm(w)
    H=np.identity(Numb)
    s=(np.identity(m)-2*np.outer(w,w))
    H[Numb-m:,Numb-m:] = s
    return H

def HQR(A):       #Implementation of the QR decomposition using Householder reflection matrices

    n=A.shape[0]
    R=A
    Q=np.matrix(np.identity(n))
    for i in range(n): 
        v=R[i:, i]
        H=Household_Matrix(v,n)
        R= np.dot(H,R)
        Q= np.dot(H.T,Q)

    return Q.T,R

def QR_eig(A):     #Implementation of Algorithm 7 p54
    L = np.tril(A, -1)
    while np.linalg.norm(L) > 10**(-25):
        Q, R = HQR(A)
        A = R @ Q
        L = np.tril(A, -1)
    
    return A.diagonal()

def lanczos(A, num):     #Implementation of the second half of page 57
    Q = arnoldi(A, num)
    H = np.dot(Q.T,np.dot(A,Q))
    eigenvalues = QR_eig(H)
    
    small_eigenvectors = []
    for i in np.nditer(eigenvalues):
        v = np.linalg.solve(H-i*np.identity(H.shape[0]),10**(-25)*np.ones(H.shape[0]))
        small_eigenvectors.append(v/np.linalg.norm(v))
    
    
    eigenvectors=np.zeros((ker.shape[0], NumberOfClusters))
    for i in range(NumberOfClusters):
        v = Q @ small_eigenvectors[i]
        eigenvectors[:,i] = v
 
        
    return eigenvalues, eigenvectors

def powermethod(A):  #Implementation of algorithm 8 p57 but with while loop instead of for loop
    v = np.random.rand(A.shape[1])
    A = np.array(A)
    diff = 10
    mu = 0
    
    while diff > 0.0000000001:
        Av = np.matmul(A, v)
        v = Av/np.linalg.norm(Av)
        mu_new = np.dot(v, Av) / np.dot(v,v)
        diff = np.abs(abs(mu)-abs(mu_new))
        mu = mu_new
        
    return v, abs(mu)

ev_max = powermethod(lap)[1]

shiftlap = lap - ev_max *np.identity(lap.shape[0])



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

minESS = 10**8
mediumlist = []

for j in range(10):
    vecs2b = lanczos(shiftlap, 50)[1]
    for i in range(20):
        print(i)
        centroids = K_means_plus_plus(NumberOfClusters, vecs2b.shape[0], vecs2b)
        cluster_allocation = K_means_clustering(NumberOfClusters, vecs2b, centroids)
        
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
            minESS = ESS
            beslist2b = clusterlist.copy()
            
        if ESS<0.22:
            mediumlist.append(clusterlist)

        if len(clusterlist[0]) > 100 and len(clusterlist[1]) > 100:
            mediumlist.append(clusterlist)
