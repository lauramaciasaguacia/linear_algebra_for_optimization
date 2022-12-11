import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from scipy.stats import norm


pd.set_option('display.max_columns', 500)
df = pd.read_csv('heart.csv', names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                                     "fasting blood sugar", "resting electrocardiographic results",
                                     "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                                     "slope of the peak exercise ST", "number of major vessels", "thalassemia",
                                     "heart disease"])
#
# X = df["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
#         "fasting blood sugar", "resting electrocardiographic results",
#         "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
#         "slope of the peak exercise ST", "number of major vessels", "thalassemia"]



def G(y, pred_y):
    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc

# def K(g,c):
#     return sklearn.svm.svc(C=c,gamma=g,kernel='rbf')

def hint(n):
    M=np.zeros([n,2])
    M[:,0] = np.random.uniform(1,10**9,n) #C first column
    M[:, 1] = np.random.uniform(10 ** -10, 1,n) #gamma second column

    return M

def MakeSigma(H):
    l=1 #how to choose ASK   #Grid search fr this hp
    HNorm = np.sum(H ** 2, axis=-1)
    Sigma = np.exp(- (HNorm[:, None] + HNorm[None, :] - 2 * np.dot(H, H.T))/(2* l**2))
    return Sigma

def MakeKappa(H,h):
    l=1 #how to choose ASK   #Grid search fr this hp
    hvec = np.tile(h, (H.shape[0], 1))
    Norm2 = np.sum(np.square(H - hvec), axis=1)

    return np.exp((-Norm2) / (2 * l ** 2))

def MakeKappaPrime(H,h):
    l=1 #how to choose ASK   #Grid search fr this hp
    K=MakeKappa(H,h)
    hvec = np.tile(h, (H.shape[0], 1))
    d=(H-hvec)/(l ** 2)
    DiagK=np.diagflat(K)

    return DiagK @ d

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
# def MuStd(H,h,y):
#
#     smallk=MakeKappa(H,h)
#     Sigma=MakeSigma(H)
#     SigmaInv=np.linalg.inv(Sigma)
#     kSigmaInv = smallk.T @ SigmaInv
#     mu= kSigmaInv @ y
#     std=np.sqrt(1-kSigmaInv @ k)
#
#     return mu, std


def GetGrad(H,h,y):

    ybest= np.max(y)

    k=MakeKappa(H,h)
    kgamma,kC = MakeKappaPrime(H,h)
    S=MakeSigma(H)

    Sinv = inverse( S ) #how to invert this

    Sy = Sinv @ y
    Sk = Sinv @ k

    mu= k.T @ Sy
    std = np.sqrt(1 - k.T @ Sk)

    mugamma = kgamma @ Sy
    muC= kC @ Sy

    stdgamma = (-1) * (kgamma.T @ Sk)/(np.sqrt( 1- k.T @ Sk))
    stdC = (-1) * (kC.T @ Sk)/(np.sqrt( 1- k.T @ Sk))

    dwrtgamma = norm.pdf((ybest - mu) / std) * (mugamma * std - (ybest - mu) * stdgamma) / std ** 2
    dwrtC = norm.pdf((ybest - mu) / std) * (muC * std - (ybest - mu) * stdC) / std ** 2

    Grad= np.matrix([dwrtgamma,dwrtC])

    return Grad


def Adam(H,y):

    m = np.zeros([1,2])
    v = np.zeros([1, 2])

    alpha= 0.001
    b1=0.9
    b2=0.999
    epsilon = np.ones([1,2])* (10 ** (-8))
    k=1

    xk=np.zeros([1,2])  #choose starting point

    tol=1

    while k <100 and tol>0.01:
        #I think we should put a minus here because we want to minimiza the NEGATIVE of the fucntion
        grad = (-1) * GetGrad(H,xk,y)

        mNew = (b1 * m + (1-b1) * grad) /(1-b1**k)
        vNew = (b2 * v + (1-b2) * np.square(grad)) / (1-b2**k)   #as suggested in the paper to compute bias corrected value


        tol = alpha * np.divide(mNew, (np.sqrt(v) + epsilon) )

        xk1= xk- tol


        m = mNew
        v = vNew

        xk=xk1

        k= k + 1

    return xk


def main():
    n=5 #chose how many starting points
    H=hint(n)

    #Blackbox funtcion using H
    y = G(results,predictions)


    for i in range(100):
        hnew = Adam(H,y)
        H=np.append(H, hnew)

        #Blackbox function using hnew
        y = np.append(y, G(results,predictions))





