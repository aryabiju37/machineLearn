import numpy as np
import matplotlib.pyplot as plt

#loading the data
X = []
Y = []
N = 0
for line in open('data_1d.csv'):
    N = N + 1
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

#turning into numpy arrays
X = np.array(X)
Y = np.array(Y)
print(N)

#plotting to visualize the data
#plt.scatter(X,Y)
#plt.show()

#apply the equations we learned to calculate a and b

#denomenator = np.dot(X,X) - [np.mean(X)*np.mean(X)]
#a = [np.dot(X,Y) - np.dot(X.mean(),Y.mean())] / denomenator
#b = [np.dot(X.mean(),Y.mean()) - (np.mean(X) * np.dot(X,Y))] / denomenator
#Xbar Ybar for single instance of a and b
#Xbar = X / N
#Ybar = Y / N
# here an array of X and Y so we take  the entire array
denomenator = X.dot(X) - X.mean() * X.sum()
a =(X.dot(Y) - Y.mean() * X.sum()) / denomenator
b =  (Y.mean()*X.dot(X)-X.mean()*X.dot(Y)) / denomenator 
#predicted y

Yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

#R2 error
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("***************")
print("the r squared is {0}",r2)
