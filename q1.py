import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

"""In this file we perform a grid search in the hyper-parameter space. You can choose the granularity of the grid and
the script will output the best combination of hyper-parameters found according to the average accuracy in the K-fold
cross validation. It will also plot this average accuracy for each combination of hyper-parameters in a heatmap."""

pd.set_option('display.max_columns', 20)

# Read the data
df = pd.read_csv('heart.csv', names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
                                     "fasting blood sugar", "resting electrocardiographic results",
                                     "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
                                     "slope of the peak exercise ST", "number of major vessels", "thalassemia",
                                     "heart disease"])

# Split into input and output
X = df[["age", "sex", "chest pain type", "resting blood pressure", "serum cholestoral",
        "fasting blood sugar", "resting electrocardiographic results",
        "maximum heart rate achieved", "exercise induced angina", "ST depression induced",
        "slope of the peak exercise ST", "number of major vessels", "thalassemia"]].values

y = df["heart disease"].values


n_C = 25  # Granularity in the C dimension
n_gamma = 25  # Granularity in the gamma dimension


C_arr = 10 ** np.linspace(0, 9, num=n_C)
gamma_arr = 10 ** np.linspace(-10, 0, num=n_gamma)

K = 5  # Number of folds for the cross validation
test_size = 270 / 5

kf = KFold(n_splits=K, shuffle=True, random_state=27)


def accuracy(y, pred_y):  # This function returns the accuracy for a given predicted and true set of datapoints
    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc


results = np.zeros((n_C, n_gamma))  # We start with an empty array of accuracies

i = 0
for C in C_arr:  # Loop through every C in the grid
    j = 0
    for gamma in gamma_arr:  # Loop through every gamma in the grid
        acc_sum = 0
        for train_index, test_index in kf.split(X):  # Loop through the splits and sum accuracy
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel='rbf'))
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy(y_test, y_pred)

            acc_sum += acc

        mean_acc = acc_sum / K  # Average accuracy

        results[i][j] = mean_acc  # Update the results array for the current combination of hyper-parameters

        j += 1

    i += 1

max_ind = np.unravel_index(results.argmax(), results.shape)  # What is the location of the best accuracy
print("best accuracy:", np.max(results))
print("best log_C:", np.linspace(0, 9, num=n_C)[max_ind[0]])
print("best log_gamma:", np.linspace(-10, 0, num=n_gamma)[max_ind[1]])

xticks = np.linspace(-10, 0, endpoint=False, num=10)[1:]
yticks = np.linspace(0, 9, endpoint=False, num=10)[1:]

# Plot the heatmap of accuracies
ax = sns.heatmap(results)
ax.set(ylabel='log_C', xlabel='log_Gamma')
ax.set_xticks(np.linspace(0, n_gamma, endpoint=False, num=10)[1:])
ax.set_yticks(np.linspace(0, n_C, endpoint=False, num=10)[1:])
ax.set_xticklabels(f'{c:.1f}' for c in xticks)
ax.set_yticklabels(f'{c:.1f}' for c in yticks)
plt.show()

