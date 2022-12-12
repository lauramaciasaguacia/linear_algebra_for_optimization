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


def accuracy(y, pred_y):
    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc



def hint(n):
    M=np.zeros([n,2])
    M[:, 0] = np.random.uniform(10 ** -10, 1,n) #gamma first column
    M[:,1] = np.random.uniform(1,10**9,n) #C second column

    return M

def MakeSigma(H):
    l=1 #how to choose ASK   #Grid search fr this hp
    HNorm = np.sum(H ** 2, axis=-1)
    Sigma = np.exp(- (HNorm[:, None] + HNorm[None, :] - 2 * np.dot(H, H.T))/(2* l**2))
    return Sigma


def MakeKappa(H,h):
    l=100000000 #how to choose ASK   #Grid search fr this hp
    hvec = np.tile(h, (H.shape[0], 1))
    Norm2 = np.sum(np.square(H - hvec), axis=1)
    f= -Norm2[:, None]/(2 * l ** 2)

    return np.exp( (f) )

def MakeKappaPrime(H,h):
    l=100000000 #how to choose ASK   #Grid search fr this hp
    K=MakeKappa(H,h)

    hvec = np.tile(h, (H.shape[0], 1))

    d=(H-hvec)/(l ** 2)

    A = K * d

    return A





