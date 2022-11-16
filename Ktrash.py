import time
import numpy as np
import pandas as pd
from tabulate import tabulate

start=time.time()


def normalize_min_max(array):  #Normalizing the dataset. All entry will be contained between 0 and 1
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))

    return array

def normalize_z_score(array):
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, i])

    return array

def cluster_start(number_of_clusters, X): #Initialize the clusters randomly
    cluster_list = []

    for i in range(number_of_clusters):
        cluster_list.append([])

    for i in range(X.shape[0]):
        assigned_list = np.random.choice(number_of_clusters)
        cluster_list[assigned_list].append(i)   #Lists of elements in each cluster
    return cluster_list

def G_kernel_matrix( X,g ):
    X_norm = np.sum(X ** 2, axis = -1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
    print(K)
    return K


def C_kernel_matrix(X, g): #Cauchy Kernel
    X_norm = np.sum(X ** 2, axis=-1)
    A = np.ones([X.shape[0],X.shape[0]])+g*(X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
    print('A',A)
    K=np.reciprocal(A)
    print('K',K)
    return K


def E_kernel_matrix(X, g): #Cauchy Kernel
    X_norm = np.sum(X ** 2, axis=-1)
    A = np.ones([X.shape[0],X.shape[0]])+g*(X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
    K=np.reciprocal(A)

    return K

def H_kernel_matrix(X,g): #HyperTangent Kernel
    X_norm = np.sum(X ** 2, axis=-1)
    K = np.ones([X.shape[0], X.shape[0]]) -  np.tanh(g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
    return K


def ESD_calculator(clusterlist,kernelmatrix,clustersum):
    ESD=0

    for k in range(NumberOfClusters):
        card = len(clusterlist[k])
        c = clustersum[k]
        for f in clusterlist[k]:
            a = kernelmatrix[f, f]
            b = kernelmatrix[f, clusterlist[k]].sum()
            dfk= a-2/card * b + c/(card**2)
            ESD=ESD + dfk

    return ESD
def gamma_parameter(array):
    av = []
    for i in range(array.shape[1]):
        av.append(np.average(array[:, i]))
    gamma = 0
    for f in range(array.shape[0]):
        gamma = gamma + np.linalg.norm(array[f, :] - av) ** 2
    gamma = array.shape[0] / gamma

    return gamma

def kernel_cluster(ker, ClusterList, NumberOfClusters):
    ##
    ClusterSum = np.zeros(NumberOfClusters)

    for k in range(NumberOfClusters):  # This is the sum of each cluster, or third term in the formula
        ClusterSum[k] = (ker[:, ClusterList[k]])[ClusterList[k], :].sum()

    ESD = ESD_calculator(ClusterList, ker, ClusterSum)
    Old_ESD=ESD+1
    z = 0
    numb_changes = 1

    while (ESD-Old_ESD)<0 and z < 100:#We can also use number of changes since it is equivalent
        numb_changes=0
        Old_ESD=ESD
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
        ESD = ESD_calculator(ClusterList, ker, ClusterSum)
        # print('ESD Variation', ESD-Old_ESD)
        # print('Number Of Changes:', numb_changes)
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





pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array2=array.copy()
# array = normalize_z_score(array)
# array = normalize_min_max(array)
array = np.delete(array, 0, 1)
array2= np.delete(array2, 0, 1)



NumberOfClusters = 4
NumberOfIterations=5
gamma= gamma_parameter(array)

ker = G_kernel_matrix(array, gamma)
ker = C_kernel_matrix(array, gamma)
ker = G_kernel_matrix(array, gamma)



Best_ESD=0
Best_cluster_list=[]
t=time.time()

cc=0
for i in range(NumberOfIterations):
    print('Checking:',i+1,'/',NumberOfIterations, 'Time:',time.time()-t)
    ClusterList = cluster_start(NumberOfClusters, array)
    ClusterSum = kernel_cluster(ker, ClusterList, NumberOfClusters)
    ESD=ESD_calculator(ClusterList,ker,ClusterSum)

    if i==0:
        Best_ESD=ESD+1

    if ESD< Best_ESD:
        Best_cluster_list=ClusterList.copy()
        Best_ESD=ESD
        cc +=1

print('Best esd', cc)
print_my_table(array2,NumberOfClusters,Best_cluster_list)

