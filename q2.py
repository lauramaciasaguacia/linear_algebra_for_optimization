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

y = df["heart disease"].values

np.random.seed(31)
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
    k=MakeKappa(H,h)
    S=MakeSigma(H)
    Sinv = np.linalg.inv( S ) #how to invert this
    Sy = np.dot(Sinv, y[:, None])
    Sk = np.dot(Sinv, k)
    mu= np.dot(k.T , Sy)[0,0]
    print("Smallest Eig of Sigma:", np.min(np.linalg.eigvals(S)))
    if np.dot(k.T, Sk)[0, 0] > 1:
        print("Okai we have a problem")
        print("Matrix H")
        print("Eigs of Sigma are:", np.linalg.eigvals(S))
        print(H)
        std = np.sqrt(abs(1 - np.dot(k.T, Sk)[0, 0]))
    else:
        std = np.sqrt(1 - np.dot(k.T, Sk)[0, 0])

    return mu,std


def GetGrad(H,h,y):

    ybest= np.max(y)
    k=MakeKappa(H,h)
    Help = MakeKappaPrime(H,h)

    kgamma= Help[:,0]
    kC = Help [:,1]

    S=MakeSigma(H)

    Sinv = np.linalg.inv( S ) #how to invert this

    Sy = np.dot(Sinv,  y[:, None])
    Sk = np.dot(Sinv, k)

    mu= np.dot(k.T , Sy)[0,0]

    if np.dot(k.T, Sk)[0,0] > 1:
        print("Okai we have a problem")
        std = np.sqrt(abs(1 - np.dot(k.T, Sk)[0, 0]))
    else:
        std = np.sqrt(1 - np.dot(k.T, Sk)[0,0])
    mugamma = np.dot(kgamma, Sy)[0]
    muC= np.dot(kC, Sy)[0]
    # print("std",std)

    stdgamma = (-1) * np.dot(kgamma.T, Sk)[0]/std
    stdC = (-1) * np.dot(kC.T,Sk)[0]/std



    dwrtgamma = norm.pdf((ybest - mu) / std) * (mugamma * std - (ybest - mu) * stdgamma) / std ** 2
    dwrtC = norm.pdf((ybest - mu) / std) * (muC * std - (ybest - mu) * stdC) / std ** 2

    Grad= np.array([dwrtgamma,dwrtC])
    return Grad



#
# def graddesc(H,y):
#     alpha= 0.01
#     xk = [-5, 4.5]  # choose starting point
#     k=0
#     tol=1
#     while k <100 and tol>0.00000001:
#         grad =  GetGrad(H,xk,y)
#         move= alpha* grad
#         xk1 = xk - move
#         print("xk desc",xk)
#         tol=np.linalg.norm(xk1-xk)
#         print("tol",tol)
#         xk=xk1
#         k+=1
#         xk = np.array(xk)
#         xk1 = np.array(xk1)
#
#
#     return xk
#

def BestAdam(H,y):
    ybest= np.max(y)
    bxk=Adam(H,y)
    mu,std = MakeMuStd(H,bxk,y)
    z= (ybest-mu)/std
    best= norm.cdf( z)
    k=0
    while k<5 and best>0.05:
        xk=Adam(H,y)
        mu, std = MakeMuStd(H, xk, y)
        z = (ybest - mu) / std
        p = norm.cdf(z)
        k+=1
        if p<best:
            bxk=xk
            best=p
            print("better minimizer of the acquistion function found k:",k,"p",p)

    print("best p", best, "k:",k)

    return bxk


def Adam(H,y):

    m = np.zeros([1,2])
    v = np.zeros([1, 2])

    alpha= 0.001
    b1=0.9
    b2=0.999
    epsilon = np.ones([1,2])* 10** -8
    k=1

    xk=[np.random.uniform(-10, 0), np.random.uniform(0, 9)]  #Randomly choose starting point

    tol=1

    while k < 50 and tol>10**-20:
        grad = GetGrad(H,xk,y)

        mNew = (b1 * m + (1-b1) * grad) /(1-b1**k)
        vNew = (b2 * v + (1-b2) * np.square(grad)) / (1-b2**k)   #as suggested in the paper to compute bias corrected value

        move = alpha * np.divide(mNew, (np.sqrt(vNew) + epsilon) )

        xk1= xk-move
        # print("move:",np.linalg.norm(move))

        m = mNew
        v = vNew


        xk = np.array(xk)
        xk1 = np.array(xk1)[0]

        if xk1[0] < -10:
            xk1[0] = -10
            # print("Gamma")

        if xk1[0] > 0:
            xk1[1] = 0
            # print("Gamma")

        if xk1[1] < 0:
            xk1[1] = 0
            # print("C")

        if xk1[1] > 9:
            xk1[1] = 9
            # print("C")

        # if np.linalg.norm(grad)< 10**-20:
        #     print("k because of Grad",k)
        #     return xk1

        tol = np.linalg.norm(xk - xk1)
        xk = xk1
        k = k + 1


    # print("k", k)
    return xk

def EvalAcc(hnew,X,y,kf):
    K = 5
    # print('hnew',hnew)
    acc_sum = 0

    gg = 10 ** hnew[0]
    CC = 10 ** hnew[1]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = make_pipeline(StandardScaler(), SVC(C=CC, gamma=gg, kernel='rbf'))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy(y_test, y_pred)
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
        mean_acc = EvalAcc(h,X,y,kf)
        z= np.append(z,mean_acc )

    for i in range(190):
        print("Iteration ", i)
        hnew = BestAdam(H,z)

        if math.isnan(hnew[0]) or math.isnan(hnew[1]):
            print("we got him")
        else:
            H=np.vstack( (H, hnew) )
            mean_acc = EvalAcc(hnew, X, y, kf)
            z= np.append(z,mean_acc )

    bestPosition = np.argmax(z)
    BestGamma,BestC = H[bestPosition,:]
    print("Best Accuracy:",z[bestPosition],"best log(g) and log(c):", BestGamma, BestC)


    imaging(H,z,n)

    return


main()





