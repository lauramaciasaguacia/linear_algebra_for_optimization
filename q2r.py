from scipy.stats import norm
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import math


pd.set_option('display.max_columns', 500)
df = pd.read_csv('heart.csv', names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                                     "fasting blood sugar", "resting electrocardiographic results",
                                     "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                                     "slope of the peak exercise ST", "number of major vessels", "thalassemia",
                                     "heart disease"])

X = df[["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
        "fasting blood sugar", "resting electrocardiographic results",
        "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
        "slope of the peak exercise ST", "number of major vessels", "thalassemia"]].values

Y = df["heart disease"].values

K = 5
test_size = 270 / 5
kf = KFold(n_splits=K, shuffle=True, random_state=27)  # Mat' bday

np.random.seed(31072000)
def accuracy(y, pred_y):
    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc

# def K(g,c):
#     return sklearn.svm.svc(C=c,gamma=g,kernel='rbf')

def hint(n):
    M=np.zeros([n,2])
    M[:, 0] = np.random.uniform(-10, 0,n)  #log(gamma) first column
    M[:,1] = np.random.uniform(0,9,n)  #log(C) second column

    return M

def MakeSigma(H):
    l=1  #how to choose ASK   #Grid search fr this hp
    HNorm = np.sum(H ** 2, axis=-1)
    Sigma = np.exp(- (HNorm[:, None] + HNorm[None, :] - 2 * np.dot(H, H.T))/(2* l**2))
    return Sigma


def MakeKappa(H,h):
    l=1 #how to choose ASK   #Grid search fr this hp
    hvec = np.tile(h, (H.shape[0], 1))
    Norm2 = np.sum(np.square(H - hvec), axis=1)
    f= Norm2[:, None]/(2 * l ** 2)

    return np.exp( -(f) )

def MakeKappaPrime(H,h):
    l=1 #how to choose ASK   #Grid search fr this hp
    K=MakeKappa(H,h)

    hvec = np.tile(h, (H.shape[0], 1))

    d=(H-hvec)/(l ** 2)

    A = K * d

    return A

# def MakeEGG(Sigma,smallk,h):
#     K = 1 #Because of the kernel we chose
#     EGG=np.zeros([Sigma.shape[0]+1,Sigma.shape[0]+1])
#     EGG[:Sigma.shape[0],:Sigma.shape[0]]= Sigma
#     EGG[0:Sigma.shape[0],Sigma.shape[0]]=smallk
#     EGG[Sigma.shape[0],0:Sigma.shape[0]]=smallk.T
#     EGG[-1,-1]=K
#
#     return EGG
#

def MakeMuStd(H,h,y):

    #INPUT: ~H: matrix containg the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #
    #OUTPUT: The average and standard deviation as calculated in the report

    k=MakeKappa(H,h)
    S=MakeSigma(H)
    Sinv = np.linalg.inv( S ) #how to invert this
    Sy = np.dot(Sinv, y[:, None])
    Sk = np.dot(Sinv, k)
    mu= np.dot(k.T , Sy)[0,0]
    if np.dot(k.T, Sk)[0, 0] > 1:
        std = 0
    else:
        std = np.sqrt(1 - np.dot(k.T, Sk)[0, 0])

    return mu,std


def GetGrad(H,h,y,Sinv):
    #INPUT: ~H: matrix containg the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #       ~Sinv: Inverse of matrix Sigma. It is passed to not calculate it multiple times
    #
    #OUTPUT: the gradient (partial derivatives) of the acquisiton function evaluated at h


    ybest= np.max(y)
    k=MakeKappa(H,h)
    Help = MakeKappaPrime(H,h)
    kgamma= Help[:,0]
    kC = Help [:,1]

    Sy = np.dot(Sinv,  y[:, None])
    Sk = np.dot(Sinv, k)

    mu= np.dot(k.T , Sy)[0,0]

    if np.dot(k.T, Sk)[0,0] > 1:
        return np.array([0,0])
    else:
        std = np.sqrt(1 - np.dot(k.T, Sk)[0,0])

    mugamma = np.dot(kgamma, Sy)[0]
    muC= np.dot(kC, Sy)[0]

    stdgamma = (-1) * np.dot(kgamma.T, Sk)[0]/std
    stdC = (-1) * np.dot(kC.T,Sk)[0]/std



    dwrtgamma = -norm.pdf((ybest - mu) / std) * (mugamma * std - (ybest - mu) * stdgamma) / std ** 2
    dwrtC = -norm.pdf((ybest - mu) / std) * (muC * std - (ybest - mu) * stdC) / std ** 2

    Grad= np.array([dwrtgamma,dwrtC])
    return Grad


def BestAdam(H,y):
    alpha= 0.01
    b1=0.9
    b2=0.999
    Sinv = np.linalg.inv(MakeSigma(H))

    bxk=Adam(H,y,Sinv,alpha,b1,b2)[0]
    best=probability(H,bxk,y)
    k=0

    while k<50 and best>0.25:

        xk=Adam(H,y,alpha,b1,b2)[0]
        p=probability(H,xk,y)
        k+=1
        if p<best:
            bxk=xk
            best=p
            # print("better minimizer of the acquistion function found k:",k,"p",p)

    print("best p", best, "k:",k)

    return bxk
def probability(H,x,y):
    ybest = np.max(y)
    mu, std = MakeMuStd(H, x, y)

    if std != 0:
        z = (ybest - mu) / std
        p = norm.cdf(z)
    else:
        p = (mu < ybest)

    return p
def toll(H,a,b,y): #conservative
    t=np.sum(np.square(a-b)) + (probability(H,a,y)-probability(H,b,y)**2)
    return t


def Adam_HO(H,y,Sinv):
    T=np.array([0,0,0,0])
    alpha_grid=[0.1,0.01,0.001]
    b1_grid=np.linspace(0.7,1,31)
    b2_grid=np.linspace(0.7,1,61)
    for a in alpha_grid:
        for b1 in b1_grid:
            for b2 in b2_grid:
                correct=0
                for i in range(100):
                    support=Adam(H,y,0,Sinv,a,b1,b2)
                    if support[2]<support[1]:
                        correct= correct+1
                T=np.vstack( (T, np.array([a,b1,b2,correct])) )
    np.savetxt('values.csv', T, delimiter=",")
    return

def Adam(H,y,Sinv,alpha,b1,b2):
    printing = 0  #Set to 1 if you want to print the descent plot

    m = np.zeros([1, 2])
    v = np.zeros([1, 2])

    k=1

    if printing:
        aux=[]

    xk=[np.random.uniform(-10, 0), np.random.uniform(0, 9)]  #Randomly choose starting point

    Initial=probability(H,xk,y)

    tol1=1

    while k < 500 and tol1>10**-6:

        grad = GetGrad(H,xk,y,Sinv)
        mNew = (b1 * m + (1 - b1) * grad)
        vNew = (b2 * v + (1 - b2) * np.square(grad))
        if any(vNew)==0:
            return xk

        move = alpha * np.divide(mNew, (np.sqrt(vNew) ))

        xk1= xk-move

        m = mNew
        v = vNew

        xk = np.array(xk)
        xk1 = np.array(xk1)[0]

        if xk1[0] < -10:
            xk1[0] = -10

        if xk1[0] > 0:
            xk1[0] = 0

        if xk1[1] < 0:
            xk1[1] = 0

        if xk1[1] > 9:
            xk1[1] = 9

        tol1 = toll(H,xk,xk1,y)

        if printing:
            aux.append(xk)

        xk = xk1

        k = k + 1

    if printing:
        descent_plot(aux,H,y)

    Final=probability(H,xk,y)

    return xk,Initial,Final

def descent_plot(aux,H,y):
    print(len(aux))
    aux = np.array(aux)
    if aux[-1,0]>aux[0,0]:
        lb1=aux[0, 0]*1.2
        ub1=aux[-1, 0] * 0.8
    else:
        lb1=aux[-1, 0] *1.2
        ub1=aux[0, 0] * 0.8
    if aux[-1, 1] > aux[0, 1]:
        lb2=aux[0, 1] *0.8
        ub2=aux[-1, 1] * 1.2
        x2 = np.linspace(aux[0, 1] *0.8, aux[-1, 1] * 1.2, 150)
    else:
        lb2=aux[-1, 1] *0.8
        ub2=aux[0, 1]*1.2

    x1 = np.linspace(lb1,ub1, 150)
    x2 = np.linspace(lb2,ub2, 150)

    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            ij = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = probability(H, ij, y)
    fig = plt.figure(figsize=(10, 7))
    plt.imshow(Z, extent=[lb1,ub1, lb2, ub2], origin='lower', cmap='viridis', alpha=1)
    plt.title("ADAM: Minimization of acquisition function", fontsize=15)
    plt.plot(aux[:, 0], aux[:, 1])
    plt.plot(aux[:, 0], aux[:, 1], '.',color='lightgray', label="Acquisition function")
    plt.plot(aux[-1, 0], aux[-1, 1],'o', color='r', label="Last Point")
    plt.plot(aux[0, 0], aux[0, 1], 's',color='fuchsia', label="First Point")

    plt.xlabel('log(gamma)', fontsize=11)
    plt.ylabel('log(C)', fontsize=11)
    plt.colorbar()
    plt.legend(loc="upper right")
    plt.show()

    return
def EvalAcc(hnew,X,Y,kf):
    K = 5
    # print('hnew',hnew)
    acc_sum = 0

    gg = 10 ** hnew[0]
    CC = 10 ** hnew[1]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        clf = make_pipeline(StandardScaler(), SVC(C=CC, gamma=gg, kernel='rbf'))
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)

        acc = accuracy(Y_test, Y_pred)
        acc_sum += acc

    mean_acc = acc_sum / K

    return mean_acc

def imaging(H,z,n):
    BestSoFar=[]
    value=z[0]
    for i in range(n,len(z)):
        if z[i]>=value:
            value=z[i]
        BestSoFar.append(value)

    plt.plot(range(n,len(z)),BestSoFar)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    col=[* 'c' * len(z)]
    for i in range(n):
        col[i]='m'
    col[np.argmax(z)]='r'

    for i in range(len(z)):
        ax.scatter(H[i,0], H[i,1], z[i],c= col[i],marker = 'o')

    ax.set_xlabel('log(gamma) Label')
    ax.set_ylabel('log(C) Label')
    ax.set_zlabel('Accuracy Label')

    plt.show()


    return


def main():
    n=5 #chose how many starting points
    H=hint(n)

    K = 5
    test_size = 270 / 5
    kf = KFold(n_splits=K,shuffle=True, random_state=27) #Mat' bday
    z=np.empty(0)

    for i in range(H.shape[0]):
        h=H[i,:]
        mean_acc = EvalAcc(h,X,Y,kf)
        z= np.append(z,mean_acc )

    # Sinv=np.linalg.inv(MakeSigma(H))
    # Adam_HO(H, z, Sinv)

    for i in range(100):
        hnew = BestAdam(H,z)
        H=np.vstack( (H, hnew) )
        mean_acc = EvalAcc(hnew, X, Y, kf)
        z= np.append(z,mean_acc )

    bestPosition = np.argmax(z)
    BestGamma,BestC = H[bestPosition,:]
    print("Best Accuracy:", z[bestPosition] , "best log(g) and log(c):", BestGamma, BestC)

    imaging(H,z,n)

    return




main()