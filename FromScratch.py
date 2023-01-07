from scipy.stats import norm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.linalg import svdvals


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


def accuracy(y, pred_y):
    # INPUT: ~y: weather heart disease is present or not
    #       ~pred_y: Whether sklearn thinks that pearson will have heart disease or not
    #
    # OUTPUT: What percentage was right

    acc = 1 - (np.sum(abs(y - pred_y) / 2) / len(y))
    return acc


def hint(n):
    # INPUT: ~n: Number of initial points of the function
    #
    #
    # OUTPUT: ~M: matrix containing the points at which to initially evaluate G

    M = np.zeros([n, 2])
    M[:, 0] = np.random.uniform(-10, 0, n)  # In the first column we have the log(gamma) coordinate
    M[:, 1] = np.random.uniform(0, 9, n)  # In the first column we have the log(C) coordinate

    return M


def MakeSigma(H, l):
    # INPUT: ~H: Matrix containing the points at which we have evaluated the
    #        ~l: length scale parameter
    #
    # OUTPUT: ~M: matrix containing the points at which to initially evaluate G

    HNorm = np.sum(H ** 2, axis=-1,dtype='float64')
    D = (HNorm[:, None] + HNorm[None, :] - 2 * np.dot(H, H.T))
    np.fill_diagonal(D, 0)
    D = np.sqrt(D)
    assist = np.sqrt(3) * D / l

    Sigma = (1 + assist) * np.exp(-assist)

    np.fill_diagonal(Sigma, 1+10**-8) #Nugget Regularization and ensuring Positive Defiteness

    return Sigma


def MakeKappa(H, h, l):
    # INPUT: ~H: Matrix containing the points at which we have evaluated the
    #        ~h: New point h=(log(gamma),log(C) )
    #        ~l: length scale parameter
    #
    # OUTPUT: ~K: Vector containing the kernel evaluated at hi and h. K(hi,h)

    # hvec = np.tile(h, (H.shape[0], 1))
    #
    # d=np.subtract(H,hvec)
    #
    # Norm2 = np.sum(np.square(d), axis=1)
    # K= np.exp( -Norm2[:, None]/(2 * l ** 2) )

    hvec = np.tile(h, (H.shape[0], 1))
    d = np.sqrt(np.sum(np.square(H - hvec), axis=1))
    assist = np.sqrt(3) * d / l
    K = (1 + assist) * np.exp(-assist)

    return K


def MakeKappaPrime(H, h, l):
    # INPUT: ~H: Matrix containing the points at which we have evaluated the
    #        ~h: New point h=(log(gamma),log(C) )
    #        ~l: length scale parameter
    #
    # OUTPUT: ~Kprime: Vector containing the gradient of the kernel on the first column wrt to log(gamma) the second wrt to log(C)

    # K=MakeKappa(H,h,l)
    # hvec = np.tile(h, (H.shape[0], 1))
    # d=np.subtract(H,hvec)/(l ** 2)
    #
    # Kprime = K * d

    hvec = np.tile(h, (H.shape[0], 1))
    difference = np.subtract(hvec, H)
    d = np.sqrt(np.sum(np.square(difference), axis=1))
    assist = np.sqrt(3) * d / l

    Kgamma, KC = -(3 / (l ** 2)) * np.exp(-assist) * difference[:, 0], -(3 / (l ** 2)) * np.exp(-assist) * difference[:,
                                                                                                           1]

    return Kgamma, KC


def MakeMuStd(H, h, y, l):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #
    # OUTPUT: ~mu,std: The average and standard deviation as calculated in the report

    k = MakeKappa(H, h, l)
    S = MakeSigma(H, l)


    #FOR FASTER IMPLEMENTATION:

    #COMMENT THIS LINE
    Sk = Chol_Solve(S,k)


    #UNCOMMENT THIS LINE
    # Sk = np.linalg.solve(S,k)

    var = 1 - np.dot(k.T, Sk)

    #COMMENT THIS LINE
    mu = np.dot(k.T, Chol_Solve(S,y))

    #UNCOMMENT THIS LINE
    # mu = np.dot(k.T, np.linalg.solve(S,y))


    std = np.sqrt(var)

    return mu, std


def GetGrad(H, h, y, l,a_type):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #
    # OUTPUT: the gradient (partial derivatives) of the acquisition function evaluated at h

    ybest = np.max(y)

    mu, std = MakeMuStd(H, h, y, l)

    Sigma = MakeSigma(H, l)

    k = MakeKappa(H, h, l)

    kgamma, kC = MakeKappaPrime(H, h, l)


    #FOR FASTER RESULTS:

    #COMMENT THESE LINES
    Sy = Chol_Solve(Sigma,y)
    Sk = Chol_Solve(Sigma,k)


    #UNCOMMENT THESE
    # Sy = np.linalg.solve(Sigma,y)
    # Sk = np.linalg.solve(Sigma,k)

    mugamma, muC = np.dot(kgamma, Sy), np.dot(kC, Sy)

    stdgamma, stdC = (-np.dot(kgamma, Sk) / std), (-np.dot(kC, Sk) / std)



    if a_type=="P":
        pdf = norm.pdf((ybest - mu) / std)
        dwrtgamma = pdf * (-mugamma * std - (ybest-mu) * stdgamma) / (std ** 2)
        dwrtC = pdf * (-muC * std - (ybest- mu ) * stdC) / (std ** 2)

    elif a_type=="EI":
        cdf=norm.cdf( (mu-ybest)/std )
        pdf = norm.pdf((ybest-mu) / std)
        dwrtgamma = - ( mugamma * cdf + stdgamma * pdf )
        dwrtC = -( muC * cdf + stdC * pdf )

    return dwrtgamma, dwrtC


def Eval_probability(H, h, y, l):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #
    #
    # OUTPUT: Evaluation of the probability a.f.

    ybest = np.max(y)

    mu, std = MakeMuStd(H, h, y, l)

    z = (ybest - mu) / std
    p = norm.cdf(z)


    return p

def Eval_EI(H, h, y, l):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #
    # OUTPUT: Evaluation of the EI acquisition function

    ybest = np.max(y)

    mu, std = MakeMuStd(H, h, y, l)
    cdf = norm.cdf((mu-ybest) / std)
    pdf = norm.pdf((ybest-mu) / std)
    ei=(mu-ybest)*cdf+std*pdf

    return ei


def ADAM_Optimizer(xt,H, y, l,a_type,printing):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~h: the point at which we want to calculate the gradient
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #        ~printing: Whether to print the descent plot or not. Greatly increases computational and memory cost
    #
    # OUTPUT: The coordinates of the minimum (possibly local) of the acquisition function

    alpha = 0.001
    b1 = 0.9
    b2 = 0.999
    epsilon = 10 ** -8

    m = np.zeros_like(xt)
    v = np.zeros_like(xt)
    t = 0

    sc = 1


    if printing:
        traj = np.empty(0)
        traj = np.append(traj, xt)

    while sc and t<5000:
        t += 1
        g = np.array(GetGrad(H, xt, y, l,a_type))

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * np.square(g)

        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)

        xtt = xt - alpha * np.divide(mhat, np.sqrt(vhat) + epsilon)
        xtt = np.array(xtt)

        # If the point "runs off" the boundaries then we project it back in
        if xtt[0] < -10:
            xtt[0] = -10

        elif xtt[0] > 0:
            xtt[0] = 0

        if xtt[1] < 0:
            xtt[1] = 0

        elif xtt[1] > 9:
            xtt[1] = 9

        sc = Stopping_Criterion(xt, xtt)
        xt = xtt

        if printing:
            traj = np.vstack((traj, xt))



    if printing:
        descent_plot(traj,H,y,l,a_type)


    return xtt

def COORD_search(H,y,l,a_type,printing):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #        ~printing: Whether to print the descent plot or not. Greatly increases computational and memory cost
    #
    # OUTPUT: The coordinates of the gridsearch minimum, it is then fed into the ADAM optimizer

    n=25 #disdcussed in the report
    x1 = np.linspace(-9,0, n)
    x2 = np.linspace(0,10, n)

    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    if a_type == "P":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_probability(H, ij, y, l)
        MIN=np.unravel_index(Z.argmin(), Z.shape)
        xt=X1[MIN],X2[MIN]



    elif a_type == "EI":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_EI(H, ij, y, l)
        MAX=np.unravel_index(Z.argmax(), Z.shape)
        xt = X1[MAX], X2[MAX]

    if printing:
        Landscape(H,y,l,a_type,0,n)

    return np.array(xt)


def Stopping_Criterion(x1, x2):
    # INPUT: ~x1: Iteration i of the Adam optimizer
    #        ~x1: Iteration i+1
    #
    # OUTPUT: Whether the stopping criterion is met

    if np.sum(np.square(x1 - x2)) < 10 ** -8:
        return 0
    else:
        return 1



def Hybrid_Optimizer(H, y, l,a_type,step):
    # INPUT: ~H: matrix containing the previously evaluated points
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #        ~step: Whether to print the descent plot or not. Greatly increases computational and memory cost
    #
    # OUTPUT: The coordinates of the minimum (possibly local) of the acquisition function found

    print=0 #Do we want to print the descent plot?
    if not(step): #Exploration case
        x1 = np.random.uniform(-10, 0), np.random.uniform(0, 9)
        x1 = np.array(x1)
        return x1

    else: #Exploitation
        x1 = COORD_search(H,y,l,a_type,print)
        if a_type=="P":
            if Eval_probability(H,x1,y,l)<10**-8: #already minimum, no need for adam
                return x1


        hbest = ADAM_Optimizer(x1,H, y, l,a_type,print)

    return hbest


def Chol_Solve(A, y):
    # INPUT: ~A: matrix
    #        ~y: vector
    # OUTPUT: ~x: x solves the system Ax=y

    L = np.zeros_like(A)
    n = A.shape[0]

    for i in range(A.shape[0]):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - np.dot(L[i, 0:j], L[j, 0:j]))
            else:
                L[i, j] = (1.0 / L[j, j]) * (A[i, j] - np.dot(L[i, 0:j], L[j, 0:j]))

    z = np.zeros_like(y)
    F = L
    for i in range(n):
        z[i] = np.divide((y[i] - np.dot(F[i, :], z[:])), F[i, i])

    x = np.zeros_like(z)
    B = L.T
    for i in reversed(range(n)):
        x[i] = np.divide((z[i] - np.dot(B[i, i:], x[i:])), B[i, i])

    return x


def descent_plot(traj, H, y, l,a_type):
    # INPUT:~traj: The coordinates of points evaluated in the ADAM descent
    #       ~H: matrix containing the previously evaluated points
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #
    #
    # OUTPUT: Plots the path taken by the minimizer in two ways: Over heatmap and 3d plot

    traj = np.array(traj)

    if traj[-1, 0] > traj[0, 0]:
        lb1 = traj[0, 0] * 1.2
        ub1 = traj[-1, 0] * 0.8
    else:
        lb1 = traj[-1, 0] * 1.2
        ub1 = traj[0, 0] * 0.8
    if traj[-1, 1] > traj[0, 1]:
        lb2 = traj[0, 1] * 0.8
        ub2 = traj[-1, 1] * 1.2
    else:
        lb2 = traj[-1, 1] * 0.8
        ub2 = traj[0, 1] * 1.2

    x1 = np.linspace(lb1, ub1, 150)
    x2 = np.linspace(lb2, ub2, 150)

    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    if a_type=="P":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_probability(H, ij, y, l)
    elif a_type=="EI":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_EI(H, ij, y, l)
    fig = plt.figure(figsize=(10, 7))
    plt.imshow(Z, extent=[lb1, ub1, lb2, ub2], origin='lower', cmap='viridis', alpha=1)
    if a_type=="EI":
        plt.title("ADAM: Maximization of Expected Improvement A.F.", fontsize=15)
    if a_type == "P":
        plt.title("ADAM: Minimization of Probability A.F.", fontsize=15)

    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(traj[:, 0], traj[:, 1], '.', color='lightgray', label="Acquisition function")
    plt.plot(traj[-1, 0], traj[-1, 1], 'o', color='r', label="Last Point")
    plt.plot(traj[0, 0], traj[0, 1], 's', color='fuchsia', label="First Point")

    plt.xlabel('log(gamma)', fontsize=11)
    plt.ylabel('log(C)', fontsize=11)
    plt.colorbar()
    plt.legend(loc="upper right")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    z = []
    if a_type=="P":
        for i in traj:
            z.append(Eval_probability(H, i, y, l))

    if a_type=="EI":
        for i in traj:
            z.append(Eval_EI(H, i, y, l))

    z = np.array(z)

    ax.scatter(traj[1:len(z) - 1, 0], traj[1:len(z) - 1, 1], z[1:len(z) - 1], label="Steps")
    ax.scatter(traj[0, 0], traj[0, 1], z[0], 's', color='fuchsia', label="First Point")
    ax.scatter(traj[-1, 0], traj[-1, 1], z[-1], 'o', color='r', label="Last Point")


    if a_type=="EI":
        plt.title("ADAM: Maximization of Expected Improvement A.F.", fontsize=15)
    if a_type == "P":
        plt.title("ADAM: Minimization of Probability A.F.", fontsize=15)

    ax.set_xlabel("log(gamma)")
    ax.set_ylabel("log(C)")

    if a_type=="P":
        ax.set_zlabel("P ( G(h)< ybest )")
    elif a_type=="EI":
        ax.set_zlabel("Expected Improvement")

    plt.legend(loc="upper right")
    plt.show()

    return


def EvalAcc(h, X, Y, kf):
    # INPUT:~h: the point at which we want to calculate the accuracy (it is passed in log scale)
    #       ~X: Dataset excluding heartdisease
    #       ~Y: Heart Disease
    #       ~kf: K folds
    #
    #
    # OUTPUT: Accuracy of the blackbox function at the given h

    K = 5  # Number of fold verifications
    acc_sum = 0

    # Converting back to "normal scale" from log scale
    gg = 10 ** h[0]
    CC = 10 ** h[1]

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

def Landscape(H,y,l,a_type,step,n):
    # INPUT:~H: matrix containing the previously evaluated points
    #       ~y: the evaluation of G at the previous points stored in H
    #        ~l: length scale parameter
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement
    #      ~step: step number
    #       ~n: desired coarseness of the plot
    #
    # OUTPUT: Plots the landscape

    x1 = np.linspace(-9,0, n)
    x2 = np.linspace(0,10, n)

    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)


    if a_type == "P":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_probability(H, ij, y, l)


    if a_type == "EI":
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                ij = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = Eval_EI(H, ij, y, l)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=True)


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('log(gamma) ')
    ax.set_ylabel('log(C) ')
    ax.set_zlabel('Acquisition Function')
    if a_type=="P":
        plt.title(f"Landscape of the Probability A.F., l={l},Step={step}")
        # plt.title("Probability A.F. Hybrid Minimizer: Coarse Grid Search")
    if a_type=="EI":
        plt.title(f"Landscape of the Expected Improvement A.F., l={l},Step={step}")
        # plt.title("Expected Improvement A.F. Hybrid Minimizer: Coarse Grid Search")

    plt.show()


    return



def imaging(H, y, n,a_type):
    # INPUT:~H: matrix containing the previously evaluated points
    #       ~y: the evaluation of G at the previous points stored in H
    #       ~n: Number of initial points so to color them differently
    #       ~a_type: acquisition Function type. P=Probability, EI=Expected Improvement    #
    #
    # OUTPUT: Plots the best accuracy vs iterations and a scatterplot of the points analysed

    grid = []
    BestSoFar = []
    value = y[0]

    for i in range(n, len(y)):
        if y[i] >= value:
            value = y[i]
        BestSoFar.append(value)
        grid.append(0.863333333)

    fig = plt.figure()
    plt.plot(range(n, len(y)), BestSoFar)
    plt.plot(range(n, len(y)), grid, 'r')
    plt.xlabel("Iterations")
    plt.ylabel("Best Average Accuracy")
    if a_type=="P":
        plt.title("Probability A.F.: Best Average Accuracy over iterations")
    if a_type == "EI":
        plt.title("Expected Improvement A.F.: Best Average Accuracy over iterations")

    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    col = [*'c' * len(y)]
    for i in range(n):
        col[i] = 'm'
    col[np.argmax(y)] = 'r'

    for i in range(len(y)):
        ax.scatter(H[i, 0], H[i, 1], y[i], c=col[i], marker='o')

    m_patch = mpatches.Patch(color='m', label="Initial Point")
    c_patch = mpatches.Patch(color='c', label="BO Iteration")
    b_patch = mpatches.Patch(color='r', label="Best Point")



    ax.set_xlabel('log(gamma) ')
    ax.set_ylabel('log(C) ')
    ax.set_zlabel('Accuracy ')
    if a_type=="P":
        plt.title("Probability A.F.: Scatterplot of sampled points")
    elif a_type=="EI":
        plt.title("Expected Improvement A.F.: Scatterplot of sampled points")

    plt.legend(handles=[m_patch, c_patch,b_patch])


    return

def erank(A):
    # INPUT:~A:Matrix
    #
    # OUTPUT: effective rank of the matrix

    s=svdvals(A)
    p=s/np.sum(s)
    H=0
    for i in range(A.shape[0]):
        if p[i]!=0:
            H+=  p[i]*np.log(p[i])

    return np.exp(-H)



def Bayesian_Optimizer(seme):
    # INPUT:~Seme: setting the random seed
    #
    #
    # OUTPUT: Does most of the work

    n = 5  # chose how many starting points
    explore=8 #How much exploration
    a_type="P" #A.F. type
    l=0.5 #Length Scale Parameter
    np.random.seed(seme) #set random seed

    H = hint(n) #Initial points

    K = 5  # number of folds
    kf = KFold(n_splits=K, shuffle=True, random_state=27)  # Random State so that the folds are not generated as 1-54, 55-109..... and so on. The number is Matthias' birthday
    y = np.empty(0)

    for i in range(H.shape[0]):
        h = H[i, :]
        mean_acc = EvalAcc(h, X, Y, kf)
        y = np.append(y, mean_acc)


    for i in range(100):
        print("Iteration: ",i)
        hnew = Hybrid_Optimizer(H, y, l,a_type,i%explore)
        H = np.vstack((H, hnew))
        mean_acc = EvalAcc(hnew, X, Y, kf)
        y = np.append(y, mean_acc)
        #
        # if i==10 or i==50 or i==95:
        #     Landscape(H, y, l, a_type,i,150)



    bestPosition = np.argmax(y)
    BestGamma, BestC = H[bestPosition, :]
    print("Best Accuracy:", y[bestPosition] , "Best log(g) and log(c):", BestGamma, BestC)

    S=MakeSigma(H,l)

    plt.imshow(S)
    plt.title(f"Covariance Matrix with l={l}")
    plt.show()

    imaging(H,y,n,a_type)
    plt.show()

    return



seme = 222 #Francesco's favourite Number
Bayesian_Optimizer(seme)
