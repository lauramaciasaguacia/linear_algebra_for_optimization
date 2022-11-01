import pandas as pd
import numpy as np
import time
from tabulate import tabulate
start=time.time()

def normalize(X): #Normalize the data
    backtrack=[]
    for i in range(X.shape[1]):
        val_range = np.max(X[:, i]) - np.min(X[:, i])
        if val_range != 0:
            X[:, i] = (X[:, i] - np.min(X[:, i])) / val_range
        backtrack.append([val_range,np.min(X[:,i])])
    return X,backtrack

def cluster_start(number_of_clusters, data): #Initialize the clusters randomly
    #Im not very good with lists so thre is probably something here that is not correct but everything works in the end
    cluster_list = []
    cluster_size = np.zeros(number_of_clusters)
    for i in range(number_of_clusters):
        cluster_list.append([])

    for i in range(data.shape[0]):
        assigned_list = np.random.choice(number_of_clusters)
        cluster_list[assigned_list].append(i)   #Lists of elements in each cluster
        cluster_size[assigned_list] = cluster_size[assigned_list] + 1 #Array containing the number of elements in each cluster

    return cluster_list,cluster_size


def kernel_matrix( X ):
    g = 1 #Hyperparameter to choose
    X_norm = np.sum(X ** 2, axis = -1)
    K = np.exp(-g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))) #This forumla helped so much, Time was aroun 90 seconds earlier now half a second. Not super precise though(maybe we need to play with the data types)
    for i in range(X.shape[0]): #doing this so thta we get 1 on the diagonals instead of 0.99998
        K[i,i]=1
    return K

def un_normalize(X,back_info):
    for i in range(X.shape[1]):
            X[:, i] = (X[:, i] * back_info[i][0]) + back_info[i][1]
    return X

def summarize(X,cluster):
    summary=np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        summary[i]=np.average(X [cluster,i])
    return summary



pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array = np.delete(array, 0, 1)
aux= normalize(array)
array=aux[0]
normalization_info=aux[1]
NumberOfClusters = 4    #Decide number of clusters


aux = cluster_start(NumberOfClusters, array) #again i dont know how to return two things
ClusterList = aux[0]
ClusterSize = np.array(aux[1],int)
ker = np.matrix(kernel_matrix(array))

ClusterSum=np.zeros(NumberOfClusters)
ClusterNorm=np.zeros(NumberOfClusters)
for k in range(NumberOfClusters): #This is the sum of each cluster, or third term in the formula
    ClusterSum[k]= (ker[:, ClusterList[k]])[ClusterList[k], :].sum()
    ClusterNorm[k]=ClusterSum[k] / (ClusterSize[k]**2)

z=0
delta=1
while delta>0 and z<10:
    iter_time=time.time()
    PreviousNorm =np.array( ClusterNorm )

    for f in range(ker.shape[0]):
        a = ker[f, f]
        B=[]
        Dist = [] #List where we
        flag=0
        for i in range(NumberOfClusters): #I could implement this better because like this we tka ea vector out and possibly put it back in the same list
            if (f in ClusterList[i]) and (ClusterSize[i] != 1): #we dont want to work with empty lists
                ClusterList[i].remove(f)
                ClusterSize[i] = ClusterSize[i] - 1
                flag=1
                ex_cluster=i
        if flag==1:
            for k in range(NumberOfClusters):
                card = ClusterSize[k] #cardinality of cluster k
                b = ker[f, ClusterList[k]].sum()  #second term in the formula to minimize. sum of the kernels (fixed point, in the kluster)

                if ex_cluster==k:
                    ClusterSum[k]= ClusterSum[k]-2*b-a
# calculating c everytime was expensive, this way we use the fact that the kernel is symmetric and make it a lot faster
                c=ClusterSum[k]   #third term
                To_min = a - (2/card) * b + c/ (card ** 2)
                B.append(b) #we want to keep a list of the second terms so that we can fix cluster sum using the symmetric trick(like line 76)
                Dist.append(To_min)
            Closest_Cluster = np.argmin(Dist)
            ClusterList[Closest_Cluster].append(f)
            ClusterSize[Closest_Cluster] = ClusterSize[Closest_Cluster] + 1
            ClusterSum[Closest_Cluster]= ClusterSum[Closest_Cluster] + 2*B[Closest_Cluster] + a  #using the symmetry of the kernel
    print('----Iteration',z+1,'Time:------',time.time()-iter_time)


    for k in range(NumberOfClusters):
        ClusterNorm[k] = ClusterSum[k]  / (ClusterSize[k] ** 2)

    delta=(ClusterNorm-PreviousNorm).sum()  #Not sure if its a proper criteria
    z = z+1

print('----Total Run Time:------', time.time()-start)
print('------------------------------*********************************------------------------------')
head=["Balance","Qual_miles","cc1_miles","cc2_miles","cc3_miles","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12","Days_since_enroll","Award?","Cluster Size", "Cluster Norm squared"]
array=un_normalize(array,normalization_info)
cluster_summary=[]
for i in range(NumberOfClusters): #Again, im ont good with lists arrays and whatever
    s = summarize(array, ClusterList[i]).tolist()
    s.append(ClusterSize[i].tolist())
    s.append(ClusterNorm[i].tolist())
    cluster_summary.append(s)

print(tabulate(cluster_summary, headers=head,tablefmt="github")) #I dont know whats wrongg here but works


