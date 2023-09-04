import numpy as np
import matplotlib.pyplot as plt

#load the data
X = []
Y = []
for line in open("data_poly.csv"):
    x,y = line.split(",")
    x = float(x)
    X.append([1,x,x*x])
    Y.append(float(y))

#converting into numpy arrays
X = np.array(X)
Y = np.array(Y)

#let us plot the data 
# plt.scatter(X[:,1] , Y)
# plt.show()

#calculate weights
#same as multiple Linear Regression,only that input X changes
W = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
yhat = np.dot(X,W)

#plot it all together
plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]),sorted(yhat))
plt.show()