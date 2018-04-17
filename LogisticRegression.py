import numpy as np

#Logistic Regression class
class LogisticRegression(object):

    # constructor for this class
    def __init__(self, n_iter=1000, eta=0.3):
        self.n_iter = n_iter
        self.eta = eta
        self.beta = []
        self.grad_L = []

    #Gradient Assent function
    def grad_assent(self, X, y):

        X = self.add_column_with_ones(X)

        #Initialize beta with zeros
        self.beta = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            #Gradient asscent and
            self.beta -= self.eta * np.dot(X.T, self.likelyhood_beta(X, y)) / len(y)

            #Summation of all values
            self.grad_L.append(np.absolute(self.likelyhood_beta(X, y)).sum())
        print("Max beta", self.beta)

    #Function to calculate likelyhood of beta
    def likelyhood_beta(self, X, y):
        return self.hypothesis(X)- y

    #Function to calculate the Bernoulli Equation
    def hypothesis(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.beta)))

    #Function to predict chances of survival
    def predict(self, X):
        X = self.add_column_with_ones(X)
        return np.where(self.hypothesis(X) >= 0.5, 1, 0)

    #Function to add values of X with 1
    def add_column_with_ones(self, X):
        return np.concatenate([np.ones((len(X), 1)), X], axis=1)