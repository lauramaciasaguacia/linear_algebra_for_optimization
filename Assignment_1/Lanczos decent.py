# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:49:31 2022

@author: Wim
"""

import numpy as np
import pandas as pd
import scipy

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
# array = array[0:100]
# print(array.shape)
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

def arnoldi(A, num):
    q=np.zeros([A.shape[1],num+1])
    b=np.zeros(num+1)
    b[0]=0
    v=np.random.rand(A.shape[0])
    q[:,0]=np.zeros(len(v))
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

def Household_Matrix(v,Numb):
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
    w =  w/get_norm(w)
    H=np.identity(Numb)
    s=(np.identity(m)-2*np.outer(w,w))
    H[Numb-m:,Numb-m:] = s
    return H

def HQR(A):

    n=A.shape[0]
    R=A
    Q=np.matrix(np.identity(n))
    for i in range(n): ####
        v=R[i:, i]
        H=Household_Matrix(v,n)
        R= np.dot(H,R)
        Q= np.dot(H.T,Q)

    return Q.T,R

def QR_eig(A):
    L = np.tril(A, -1)
    while np.linalg.norm(L) > 10**(-25):
        Q, R = HQR(A)
        A = R @ Q
        L = np.tril(A, -1)
    
    return A.diagonal()

def lanczos(A, num):
    Q = arnoldi(A, num)
    H = np.dot(Q.T,np.dot(A,Q))
    eigenvalues = QR_eig(H)
    
    small_eigenvectors = []
    print('before for')
    for i in np.nditer(eigenvalues):
        v = np.linalg.solve(H-i*np.identity(H.shape[0]),10**(-25)*np.ones(H.shape[0]))
        small_eigenvectors.append(v/np.linalg.norm(v))
        print('in for')
    
    print('after for')
    
    eigenvectors=[]
    for i in small_eigenvectors:
        v = Q @ i
        eigenvectors.append(v)

    eigenvectors = np.array(eigenvectors)
        
    return eigenvalues, eigenvectors
    

print('begin')

ev_max = lanczos(lap, 30)[0][0,0]

shiftlap = lap - 2*ev_max *np.identity(lap.shape[0])

shiftev, eigenvectors = lanczos(shiftlap, 50)
unshiftev = shiftev + 2*ev_max
print(unshiftev)
print(eigenvectors)
print(eigenvectors.shape)
scipy_evs = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(lap), k=NumberOfClusters, which = 'SM')[1]
scipy_evs = scipy_evs.T

calculated_evs = eigenvectors[0:NumberOfClusters, 0]

np.set_printoptions(threshold=4000)
print(calculated_evs)
print(scipy_evs)

# print(scipy_evs[:, None])
# print(calculated_evs[:, None])

print(calculated_evs.shape)
print(scipy_evs.shape)

def angle(vector1, vector2):
    # cos(theta) = v1 dot v2 / ||v1|| * ||v2||
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    x = numerator / denominator if denominator else 0
    return np.arccos(x)

arr = []
for i in range(NumberOfClusters):
    li = []
    for j in range(NumberOfClusters):
        a = calculated_evs[i]
        b = scipy_evs[j]
        li.append(angle(a, b))

    arr.append(li)

print(np.array(arr))




