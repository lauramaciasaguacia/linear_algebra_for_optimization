import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


pd.set_option('display.max_columns', 500)
df = pd.read_csv('heart.csv', names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                                     "fasting blood sugar", "resting electrocardiographic results",
                                     "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                                     "slope of the peak exercise ST", "number of major vessels", "thalassemia",
                                     "heart disease"])

X = df["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                        "fasting blood sugar", "resting electrocardiographic results",
                        "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                        "slope of the peak exercise ST", "number of major vessels", "thalassemia"]
y = df["heart disease"]

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)



