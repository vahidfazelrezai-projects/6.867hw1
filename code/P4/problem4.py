import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pdb
import random
from sklearn import linear_model

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = np.array(data[0:1])[0]
    Y = np.array(data[1:2])[0]
    return X, Y

def lassoTrainData():
    return getData('P4/lasso_train.txt')

def lassoValData():
    return getData('P4/lasso_validate.txt')

def lassoTestData():
    return getData('P4/lasso_test.txt')

def getTrueWeights():
    data = pl.loadtxt('P4/lasso_true_w.txt')
    return np.array(data)

Xtrain, Ytrain = lassoTrainData()
Xval, Yval = lassoValData()
Xtest, Ytest = lassoTestData()
xmin = min(min(Xtrain), min(Xval), min(Xtest))
xmax = max(max(Xtrain), max(Xval), max(Xtest))


def phi(x):
    out = [x]
    for i in range(12):
        out.append(np.sin((i+1) * 0.4 * math.pi * x))
    return np.array(out)

def plot_curve(weights, label, width=1):
    x = [i/100.0 for i in range(100 * int(math.floor(xmin)), 100 * int(math.ceil(xmax)))]
    y = [np.dot(weights, phi(t)) for t in x]
    plt.plot(x, y, lw=width, label=label)

def plot_curve_fn(fn, label, width=1):
    x = [i/100.0 for i in range(100 * int(math.floor(xmin)), 100 * int(math.ceil(xmax)))]
    y = [fn(phi(t)) for t in x]
    plt.plot(x, y, lw=width, label=label)

def regress(X, Y):
    Z = np.array([phi(x) for x in X])
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z), Z)), np.transpose(Z)), Y);

def ridge(X, Y, lam):
    Z = np.array([phi(x) for x in X])
    return np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(Z), Z), lam * np.identity(Z.shape[1]))), np.transpose(Z)), Y)

def sse(X, Y, weights):
    return np.linalg.norm(Y - np.dot(weights, phi(X))) ** 2


reg2 = linear_model.Lasso(alpha = 0, fit_intercept=False)
reg = linear_model.Lasso(alpha = 0.03, fit_intercept=False)
Z = np.transpose(phi(Xtrain))
reg.fit(Z, Ytrain)
reg2.fit(Z, Ytrain)
def unit_vector(i):
    vec = [0] * 13
    vec[i] = 1
    return vec
Wtrue = getTrueWeights()
Wridge = ridge(Xtrain, Ytrain, 0.0000000000001)
Wlambda = [float(reg2.predict(unit_vector(i))) for i in range(13)]
Wlasso = [float(reg.predict(unit_vector(i))) for i in range(13)]

def plot():
    plt.plot(Xtrain, Ytrain, 'o', label="training")
    plt.plot(Xtest, Ytest, 'o', label="test")
    plt.plot(Xval, Yval, 'o', label="validation")
    plot_curve(Wtrue, 'true')
    plot_curve(Wridge, 'ridge')
    plot_curve(Wlasso, 'lasso')
    plot_curve(Wlambda, 'lambda=0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Various Regression Methods')
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.)
    plt.show()

plot()
