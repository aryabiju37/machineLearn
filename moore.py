import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')
for line in open("moore.csv"):
    r = line.split("\t")
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

# plt.scatter(X,Y)
# plt.show()
X = np.array(X)
Y = np.array(Y)
denomenator = X.dot(X) - X.mean() * X.sum()
a =(X.dot(Y) - Y.mean() * X.sum()) / denomenator
b =  (Y.mean()*X.dot(X)-X.mean()*X.dot(Y)) / denomenator

#predicted y

Yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X,Yhat)
# plt.show()

#R2 error
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("***************")
print("the r squared is {0}",r2)