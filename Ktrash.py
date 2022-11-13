import time
import numpy as np
import pandas as pd
from tabulate import tabulate

start=time.time()
np.random.seed(8)

pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')

array = df.to_numpy(dtype=np.float32)
array2=array.copy()

def normalize_min_max(array):
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

    return array

def normalize_z_score(array):
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, i])

    return array


# array = normalize_z_score(array)
array = normalize_min_max(array)
array = np.delete(array, 0, 1)
array2= np.delete(array2, 0, 1)

def cluster_start(number_of_clusters, X): #Initialize the clusters randomly
    cluster_list = []

    for i in range(number_of_clusters):
        cluster_list.append([])

    for i in range(X.shape[0]):
        assigned_list = np.random.choice(number_of_clusters)
        cluster_list[assigned_list].append(i)   #Lists of elements in each cluster
    return cluster_list

def kernel_matrix( X,g ):
    #Hyperparameter to choose
    X_norm = np.sum(X ** 2, axis = -1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))) #This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
    # for i in range(X.shape[0]): #doing this so thta we get 1 on the diagonals instead of 0.99998
    #     K[i,i]=1
    return K

def un_normalize(X,back_info):

    for i in range(X.shape[1]):
            X[:, i] = (X[:, i] * back_info[i][0]) + back_info[i][1]
    return


def meanvar(clusterlist,kernelmatrix,clustersum):
    SSE=0

    for k in range(NumberOfClusters):
        card = len(clusterlist[k])
        c = clustersum[k]
        for f in clusterlist[k]:
            a = kernelmatrix[f, f]
            b = kernelmatrix[f, clusterlist[k]].sum()
            dfk= a-2/card * b + c/(card**2)
            SSE=SSE + dfk

    return SSE

def kernel_cluster(ker, ClusterList, NumberOfClusters):
    ##
    ClusterSum = np.zeros(NumberOfClusters)

    for k in range(NumberOfClusters):  # This is the sum of each cluster, or third term in the formula
        ClusterSum[k] = (ker[:, ClusterList[k]])[ClusterList[k], :].sum()

    z = 0
    numb_changes = 1

    while numb_changes!=0 and z < 100:
        numb_changes=0
        z = 0
        for f in range(ker.shape[0]):
            a = ker[f, f]
            Dist = []  # List where we store the "distance" of point f from the Clusters
            B=[]


            for k in range(NumberOfClusters):
                if f in ClusterList[k]:
                    ex_cluster = k
                card = len(ClusterList[k])  # cardinality of cluster k
                b = ker[f, ClusterList[k]].sum()  # second term in the formula to minimize. sum of the kernels (fixed point, in the kluster)
                c = ClusterSum[k]  # third term

                To_min = a - (2 / card) * b + c / (card ** 2)
                B.append(b)  # we want to keep a list of the second terms so that we can fix cluster sum using the symmetric trick(like line 76)
                Dist.append(To_min)

            Closest_Cluster = np.argmin(Dist)
            if Closest_Cluster!=ex_cluster:
                ClusterList[ex_cluster].remove(f)
                ClusterList[Closest_Cluster].append(f)

                ClusterSum[Closest_Cluster] = ClusterSum[Closest_Cluster] + 2 * B[Closest_Cluster] + a  # using the symmetry of the kernel
                ClusterSum[ex_cluster] = ClusterSum[ex_cluster] - 2 * B[ex_cluster] + a  # using the symmetry of the kernel

            if Closest_Cluster!=ex_cluster:
                numb_changes+=1
        z = z + 1
        print('Number Of Changes:', numb_changes)
    return ClusterSum

def print_my_table(X,number_of_clusters,cluster_list):
    head = ["Balance", "Qual_miles", "cc1_miles", "cc2_miles", "cc3_miles", "Bonus_miles", "Bonus_trans",
            "Flight_miles_12mo", "Flight_trans_12", "Days_since_enroll", "Award"]
    summary = np.zeros((number_of_clusters, X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(number_of_clusters):
            summary[j, i] = np.average(X[cluster_list[j], i])
    print(tabulate(summary, headers=head, tablefmt="github"))
    return

NumberOfClusters = 4

av=[]
for i in range(array.shape[1]):
    av.append(np.average(array[:,i]))
gamma=0
for f in range(array.shape[0]):
    gamma=gamma+np.linalg.norm(array[f,:]-av) **2 #No clue how to choose this
gamma=array.shape[0]/gamma



Best_SSE=0
n=1
Best_cluster_list=[]
t=time.time()


for i in range(n):
    print('Checking:',i+1,'/',n, 'Time:',time.time()-t)
    SSE=0
    ClusterList = cluster_start(NumberOfClusters, array)
    ker = kernel_matrix(array, gamma)
    ClusterSum = kernel_cluster(ker, ClusterList, NumberOfClusters)

    SSE=meanvar(ClusterList,ker,ClusterSum)
    if i==1:
        Best_SSE=SSE+1

    if SSE< Best_SSE:
        Best_cluster_list=ClusterList.copy()
        Best_SSE=SSE




print_my_table(array2,NumberOfClusters,ClusterList)

