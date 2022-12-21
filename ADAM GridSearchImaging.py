import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_csv('values.csv',names=["a","b1","b2","perc"])
a1= df.loc[(df["a"] == 0.1),["b1","b2","perc"]]
a2= df.loc[(df["a"] == 0.01),["b1","b2","perc"]]
a3= df.loc[(df["a"] == 0.001),["b1","b2","perc"]]

a1=np.array(a1)
a2=np.array(a2)
a3=np.array(a3)

Z=np.reshape(a3[:,2],(31,61))
Z=Z[0:30,0:60] #When b1=1 or b2=1 the results are alwayss close two zero(intuitive: the moemntum never update)
b2_av=np.sum(Z,axis=0)/30
b1_av=np.sum(Z,axis=1)/60

b2=(np.linspace(0.7,0.995,60))
b1=(np.linspace(0.7,0.95,30))[:,None]

plt.plot(b2,b2_av)
plt.grid(1)
plt.xlabel("b2")
plt.ylabel("Average % arrived in a minimum")
plt.title("Average Result in function of b2 with a=0.001")
plt.show()

plt.plot(b1,b1_av)
plt.grid(1)
plt.xlabel("b1")
plt.ylabel("Average % arrived in a minimum")
plt.title("Average Result in function of b1 with a=0.001")
plt.show()



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(b2, b1, Z, edgecolor="black" ,lw=0.1, rstride=1,cmap="coolwarm", cstride=1,alpha=0.8)
ax.set(xlabel="b2",ylabel="b1",zlabel="% arrived in a minimum",title="Results of Gridsearch with a=0.001")
plt.show()

Z=np.reshape(a2[:,2],(31,61))
Z=Z[0:30,0:60] #When b1=1 or b2=1 the results are alwayss close two zero(intuitive: the moemntum never updates)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(b2, b1, Z, edgecolor="black" ,lw=0.1, rstride=1,cmap="coolwarm", cstride=1,alpha=0.8)
ax.set(xlabel="b2",ylabel="b1",zlabel="% arrived in a minimum",title="Results of Gridsearch with a=0.01")
plt.show()


Z=np.reshape(a1[:,2],(31,61))
Z=Z[0:30,0:60] #When b1=1 or b2=1 the results are alwayss close two zero(intuitive: the moemntum never updates)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(b2, b1, Z, edgecolor="black" ,lw=0.1, rstride=1,cmap="coolwarm", cstride=1,alpha=0.8)
ax.set(xlabel="b2",ylabel="b1",zlabel="% arrived in a minimum",title="Results of Gridsearch with a=0.1")
plt.show()
