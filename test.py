import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


pd.set_option('display.max_columns', 500)
df = pd.read_csv('heart.csv', names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                                     "fasting blood sugar", "resting electrocardiographic results",
                                     "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                                     "slope of the peak exercise ST", "number of major vessels", "thalassemia",
                                     "heart disease"])

print(df)

X = df["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
        "fasting blood sugar", "resting electrocardiographic results",
        "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
        "slope of the peak exercise ST", "number of major vessels", "thalassemia"]

print(X)
y = df["heart disease"]

C = 10 ** np.arange(0, 9, num=10)
gamma = 10 ** np.arange(-10, 0, num=10)

gamma = gamma[0]
C = C[0]

clf = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel='rbf'))
clf.fit(X, y)





