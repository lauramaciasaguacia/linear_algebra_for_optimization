import time
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from matplotlib import pyplot as plt



def kernel_elbow(array,NumberOfIterations,max):
    nClusters = range(1,max);
    ESD=[]
    IDs=range(array.shape[0])
    for i in nClusters:
        S=0
        print('analyzing number of clusters:',i)
        for iter in range(NumberOfIterations):
            InitialClusters = cluster_start(i,IDs)
            SD=call(i, ker,IDs,InitialClusters)[2]
            if SD<S or iter==0:
                S = SD
        ESD.append(S)
    plt.plot(nClusters, ESD,'ro',nClusters, ESD,'k')
    plt.xlabel("N Clusters")
    plt.ylabel(f"Minuimum ESD over {NumberOfIterations} random iterations")
    plt.title("Elbow method for selecting number of clusters")
    plt.show()
    return ESD

def normalize_min_max(array):  #Normalizing the dataset. All entry will be contained between 0 and 1
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.min(array[:, i])) / (np.max(array[:, i]) - np.min(array[:, i]))
    return array

def normalize_z_score(array):
    for i in range(1, array.shape[1]):
        array[:, i] = (array[:, i] - np.mean(array[:, i])) / np.std(array[:, i])
    return array

def cluster_start(number_of_clusters,IDs): #Initialize the clusters randomly
    cluster_list = []
    for i in range(number_of_clusters):
        cluster_list.append([])
    for i in IDs:
        assigned_list = np.random.choice(number_of_clusters)
        cluster_list[assigned_list].append(i)   #Lists of elements in each cluster

    return cluster_list

def Kernel_Matrix( X,g,type ):
    XNorm = np.sum(X ** 2, axis=-1)
    if type=="Gaussian":
        K = np.exp(-(g) * (XNorm[:, None] + XNorm[None, :] - 2 * np.dot(X, X.T)))

    elif type=="Cauchy":
        A = np.ones([X.shape[0], X.shape[0]]) + g*(XNorm[:, None] + XNorm[None, :] - 2 * np.dot(X, X.T))
        K = np.reciprocal(A)

    elif type=="HyperTangent":
        K = np.ones([X.shape[0], X.shape[0]]) -  np.tanh(g * (XNorm[:, None] + XNorm[None, :] - 2 * np.dot(X, X.T)))

    elif type=="Chi-Squared":
        li1 = []
        li2 = []
        for i in range(X.shape[1]):
            li1.append(np.subtract(np.repeat([X[:, i]], X.shape[0], axis=0), X[:, i][:, None]) ** 2)

        for i in range(X.shape[1]):
            x = np.add(np.repeat([X[:, i]], X.shape[0], axis=0), X[:, i][:, None]) * 0.5
            x = x + (x == 0)
            li2.append(x)

        chi_top = np.array(li1)
        chi_bottom = np.array(li2)

        K = 1 - np.sum((chi_top / chi_bottom), axis=0)

    elif type=="Polynomial":
        K= np.dot(X,X.T)
    else:
        print("No Kernel of type: {type} was found")

    return K

# def C_kernel_matrix(X): #Cauchy Kernel
#     X_norm = np.sum(X ** 2, axis=-1)
#     A = np.ones([X.shape[0],X.shape[0]])+(X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
#     K=np.reciprocal(A)
#     return K

# def H_kernel_matrix(X,g): #HyperTangent Kernel
#     X_norm = np.sum(X ** 2, axis=-1)
#     K = np.ones([X.shape[0], X.shape[0]]) -  np.tanh(g * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
#     return K

def ESD_calculator(clusterlist,kernelmatrix,clustersum,NumberOfClusters):
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
def stability(NumberOfClusters,ker,level,seed):
    np.random.seed(seed)
    IDs=range(ker.shape[0])
    InitialClusters = cluster_start(NumberOfClusters,IDs)
    ClusterList,ESD = call( NumberOfClusters, ker, IDs, InitialClusters)
    PreSet=ListToSet(ClusterList)
    HowManyToRemove=int(np.floor(level*ker.shape[0]))
    to_remove= np.random.randint( 0, ker.shape[0]+1, HowManyToRemove )
    to_remove=set(to_remove)
    IDs = list(set(IDs) - to_remove)
    for i in range(len(InitialClusters)):
        ClusterList[i]= list( set(ClusterList[i])-to_remove)

    ClusterList,UESD = call( NumberOfClusters, ker, IDs, ClusterList)
    PostSet=ListToSet(ClusterList)

    NumberOfChanges = CompareSets(PreSet,PostSet)

    S=1- NumberOfChanges/HowManyToRemove
    ER=UESD/ESD

    return S,ER

def ListToSet(L):
    S=[]
    for i in range(len(L)):
        S.append(set(L[i]))
    return S

def CompareSets(LS1,LS2):
    n=len(LS1)
    NumberOfChanges=0
    incommon=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            incommon[i,j]=len(LS1[i] & LS2[j])
            if i!=j:
                NumberOfChanges= NumberOfChanges+incommon[i,j]
    return NumberOfChanges


def gamma_parameter(array):

    gamma = 0
    for f in range(array.shape[1]):
        gamma = gamma + np.var (array[:,f ] )

    gamma = 1/(2*gamma)
    return gamma

def kernel_cluster(ker, ClusterList, NumberOfClusters,IDs):

    ClusterSum = np.zeros(NumberOfClusters)

    for k in range(NumberOfClusters):  # This is the sum of each cluster, or third term in the formula
        ClusterSum[k] = (ker[:, ClusterList[k]])[ClusterList[k], :].sum()

    ESD = ESD_calculator(ClusterList, ker, ClusterSum,NumberOfClusters)
    Old_ESD=ESD+1
    z = 0
    while ESD-Old_ESD<0 and z < 100:

        Old_ESD=ESD
        for f in IDs:
            a = ker[f, f]
            Dist = []  # List where we store the "distance" of point f from the Clusters
            B=[]


            for k in range(NumberOfClusters):
                if f in ClusterList[k]:
                    ex_cluster = k
                card = float(len(ClusterList[k])) # cardinality of cluster k
                b =ker[f, ClusterList[k]].sum()  # second term in the formula to minimize. sum of the kernels (fixed point, in the kluster)
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

        z = z + 1
        ESD = ESD_calculator(ClusterList, ker, ClusterSum,NumberOfClusters)

    return ClusterList, ClusterSum


def visualize_clusters(array,ClusterList,NumberOfClusters,head):
    colors = np.zeros(array.shape[0])
    for f in range(array.shape[0]):
        for k in range(NumberOfClusters):
            if f in ClusterList[k]:
                colors[f] = k

    for i in range(array.shape[1]):
        x = []
        w=[]
        for j in range(NumberOfClusters):
            x.append(array[ClusterList[j],i])
            w.append(np.ones(len(array[ClusterList[j],i])) / len(array[ClusterList[j],i]))
        plt.hist(x,alpha=0.5,weights=w,label=colors)
        plt.legend(loc='upper right')
        plt.title(f'Clusters on Feature: {head[i]}' )
        plt.xlabel(head[i])
        plt.ylabel('# of occurrances')
        plt.show()

    return


def scatter(array,ClusterList,NumberOfClusters,head):
    color=np.zeros(array.shape[0])
    for f in range(array.shape[0]):
        for k in range(NumberOfClusters):
            if f in ClusterList[k]:
                color[f]=k+1
    for i in range(array.shape[1]):
        for j in range(array.shape[1]):
            plt.scatter(array[:,i], array[:,j], c=color )
            plt.title(f"{head[i]} vs {head[j]}")
            plt.xlabel(head[i])
            plt.ylabel(head[j])
            plt.show()
    return

def boxplots(array,ClusterList,NumberOfClusters,head):
    for i in range(array.shape[1]):
        d = []
        for j in range(NumberOfClusters):
            d.append(array[ClusterList[j],i])
        sns.violinplot(data=d)
        # plt.legend(loc='upper right')
        plt.title(f'Clusters on Feature: {head[i]}' )
        # plt.xlabel(head[i])
        # plt.ylabel('# of occurrances')
        plt.show()

    return

def print_my_table(X,number_of_clusters,cluster_list,head):
    head.append('Size')
    summary = np.zeros((number_of_clusters, X.shape[1]+1))
    for i in range(X.shape[1]):
        for j in range(number_of_clusters):
            summary[j, i] = np.average(X[cluster_list[j], i])
    for i in range(number_of_clusters):
        summary[i,-1]=len(cluster_list[i])
    print(tabulate(summary, headers=head, tablefmt="github"))
    return

def call(NumberOfClusters,ker,IDs,InitialClusters):
    ClusterList, ClusterSum = kernel_cluster(ker, InitialClusters, NumberOfClusters,IDs)
    ESD = ESD_calculator(ClusterList, ker, ClusterSum,NumberOfClusters)

    return ClusterList,ESD








###########STEP 0: READ THE DATA AND NORMALIZING ################


pd.set_option('display.max_columns', 500)
df = pd.read_csv('EastWestAirlinesCluster.csv')
array = df.to_numpy(dtype=np.float32)
array = normalize_min_max(array)
array = np.delete(array, 0, 1)
gamma= gamma_parameter(array)


###########STEP 1: KERNEL CHOICE ################
StabilityIndex=[]
Ratio=[]
MinStabilityIndex=[]
KerTypes = ["Gaussian", "Cauchy", "HyperTangent","Chi-Squared","Polynomial"]
j=3
n=10
for i in KerTypes:
    SI=0
    ER=0
    minSI=1
    q = 0
    ker = Kernel_Matrix(array, j, i)
    print(i)
    while q<n:
        print(q)
        q += 1
        si, er = stability( j , ker , 0.1,(q+1))
        SI = SI + si
        ER = ER +er
        if si<minSI:
            minSI=si
    StabilityIndex.append(SI/n)
    Ratio.append(ER/n)
    MinStabilityIndex.append(minSI)
print(tabulate([StabilityIndex,MinStabilityIndex, Ratio], headers=KerTypes, tablefmt="github"))
print(tabulate([StabilityIndex,MinStabilityIndex, Ratio], headers=KerTypes, tablefmt="latex"))

###########STEP 2: NUMBER OF CLUSTERS CHOICE################

#
# NumberOfClusters=3
# ker = Kernel_Matrix(array, 3, "Polynomial")
#
# IDs=range(array.shape[0])
# InitialClusters=cluster_start(NumberOfClusters,IDs)
# Best_cluster_list,ESD= call(NumberOfClusters,ker,IDs,InitialClusters)
#
#
# head = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans',
#             'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award']
# # print_my_table(array,NumberOfClusters,Best_cluster_list,head)
# boxplots(array,Best_cluster_list,NumberOfClusters, head)
# # visualize_clusters(array,Best_cluster_list,NumberOfClusters, head)
# scatter(array,Best_cluster_list,NumberOfClusters,head)
