# Logistic Regression
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Loading data
dataset = pd.read_csv('titanic_data.csv')
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

# (a) Create a function that implements (2.2).
def maxBeta(X, Y, beta, N):
    sum = 0.0
    for i in N:
        a1=(1/(1+ math.exp(np.matmul(-np.transpose(beta), X[i]))))*math.exp(Y[i])

        a2 = (1-(1/(1+ math.exp(np.matmul(-np.transpose(beta), X[i])))))* math.exp(1-Y[i])
        sum += math.log(a1 + a2)
    return sum

#(b) Create a function that implements (2.3).
def grad_beta(X, Y, beta, N):
    sum = 0.0
    for i in range(N):
        sum += (1-(1/(1+ math.exp(np.matmul(-np.transpose(beta),X[i])))))*X[i]
    return sum

#(c) Create a function that implements gradient ascent.
def gradient_ascent(beta, eta, N):
    newBeta = (beta - eta * grad_beta(X, Y, beta, N))
    eta = maxBeta(X, Y, beta, N)
    for x in range(len(beta)):
        new_beta_val = beta[x] - eta
        newBeta.append(new_beta_val)
    compare= np.greater(newBeta, beta)
    if (stats.mode(compare)== "True"):
        return (beta + eta * grad_beta(X, Y, beta, N))
    else:
        eta = eta
        return (beta + eta * grad_beta(X, Y, beta, N))


# (d) Randomly split your data into training (80%) and testing (20%).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

beta= np.zeros(6)
print (beta)
for k in  range(10):
    beta= gradient_ascent(beta, 0.1, int(len(X)))
    print(beta)
print(beta)
