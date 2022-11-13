# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import scipy


### 2a - first calculate the Gaussian Kernel Matrix (Francesco's code) ###
### This code needs to be checked again, but let's assume it works ###

def kernel_matrix( X ):
    g = 1
    X_norm = np.sum(X ** 2, axis = -1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))) #This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
    return K

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
NumberOfClusters = 4    #Decide number of clusters

ker = np.matrix(kernel_matrix(array))

print(ker)

### Build a Laplacian from the Kernel Matrix  ###

def make_similarity(A, sim):
    with np.nditer(A, op_flags = ['readwrite']) as iteration: #iterate over all entries of ker with poss to rewrite
        for x in iteration:
            if x > sim:     #similarity level needs to be chosen
                x[...] = 1
            else:
                x[...] = 0
    return A

def make_diagonal(A):
    B = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        B[i, i] = np.sum(A[i,:])
    return B

similar = make_similarity(ker, 0.95)
diagonal = make_diagonal(ker)
lap = diagonal - similar

print("similarity matrix")
print(similar)
print("diagonal matrix")
print(diagonal)
print("Laplacian matrix")
print(lap)
        
vecs = scipy.sparse.linalg.eigs(lap, k=NumberOfClusters, which = 'SM')[1]
#This takes ages, so if anyone has a better alternative?


#print(vecs)
### use algorithm from part 1 to cluster the rows ###


### 2b implement a solver to find the K smallest eigenvalues ###

#Laplacian matrix is pos semi def, hence the smallest eigenvalue can be found 
#with the power iteration on ker - lambda_max I

# def powermethod(A):
#     v = np.random.rand(A.shape[1])
#     A = np.array(A)
#     diff = 10
#     mu = 0
    
#     while diff > 0.0000000001:
#         Av = np.matmul(A, v)
#         v = Av/np.linalg.norm(Av)
#         mu_new = np.dot(v, Av) / np.dot(v,v)
#         diff = np.abs(mu-mu_new)
#         mu = mu_new
        
#     return v, mu

# ev_max = powermethod(ker)[1]
# print(ev_max)

# newker = ker - ev_max * np.identity(ker.shape[0]) #make the smallest eigenvalues have the largest absolute value

# eigenvectorlist = []
# eigenvaluelist = []

# for i in range(NumberOfClusters):
#     v, mu = powermethod(newker)
#     mu = mu + ev_max
#     eigenvectorlist.append(v)
#     eigenvaluelist.append(mu)
#     newker = newker - mu * np.matmul(v, v.T)
    
# print(eigenvaluelist)


    

        
