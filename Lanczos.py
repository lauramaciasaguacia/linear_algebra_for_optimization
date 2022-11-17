import time
import numpy as np
import pandas as pd
from tabulate import tabulate


pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array = np.delete(array, 0, 1)


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


NumberOfClusters=4

A=kernel_matrix(array,gamma)

q=np.zeros([A.shape[1],NumberOfClusters+1])
b=np.zeros(NumberOfClusters+1)
b[0]=0
v=np.random.rand(A.shape[0])
q[:,0]=np.zeros(len(v))
q[:,1]= v/np.linalg.norm(v)

for k in range(1,NumberOfClusters):
    Aq=np.dot(A,q[:,k])
    a=np.dot(q[:,k].T,Aq)
    w= Aq-a*q[:,k]-b[k-1]*q[:,k-1]
    b[k]= np.linalg.norm(w)
    q[:,k+1]=(w/b[k])

Q = np.delete(q, 0, 1)
# print('Q')
# print(Q)


H= np.dot(Q.T,np.dot(A,Q))
# print('H')
# print(H)
#
#

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



