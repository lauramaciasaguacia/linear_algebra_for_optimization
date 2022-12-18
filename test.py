import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 20)
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


n_C = 5
n_gamma = 5
C_arr = 10 ** np.linspace(0, 9, num=n_C)
gamma_arr = 10 ** np.linspace(-10, 0, num=n_gamma)

K = 5
test_size = 270 / 5

kf = KFold(n_splits=K)


def accuracy(y, pred_y):
    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc


results = pd.DataFrame(index=C_arr, columns=gamma_arr)

for C in C_arr:
    for gamma in gamma_arr:
        acc_sum = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel='rbf'))
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy(y_test, y_pred)

            acc_sum += acc

        mean_acc = acc_sum / K

        results[gamma][C] = mean_acc

# xlab = ["{:.1e}".format(item) for item in gamma_arr]
# ylab = ["{:.1e}".format(item) for item in C_arr]





print(results)

s = sns.heatmap(results)
s.set(ylabel='C', xlabel='Gamma')
plt.show()






