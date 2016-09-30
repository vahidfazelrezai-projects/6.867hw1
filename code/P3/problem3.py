import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = np.array(data[0:1])[0]
    Y = np.array(data[1:2])[0]
    return X, Y

def regressAData():
    return getData('P3/regressA_train.txt')

def regressBData():
    return getData('P3/regressB_train.txt')

def validateData():
    return getData('P3/regress_validate.txt')


# returns function that evaluates as a polynomial with given weights
def poly_fn(M):
    def poly(x):
        return np.array([(x ** i) for i in range(M+1)])
    return poly

def sse(X, Y, fn, weights):
    return np.linalg.norm(Y - np.dot(weights, fn(X))) ** 2

def ridge(M, lam, X, Y):
    Xv, Yv = validateData()
    fn = poly_fn(M)
    Z = np.array([fn(x) for x in X])
    weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(Z), Z), lam * np.identity(Z.shape[1]))), np.transpose(Z)), Y)
    return sse(Xv, Yv, fn, weights)

# lams1 = [0, 0.3, 0.6, 0.9, 1.2]
# lams2 = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# X, Y = regressAData()
# lams = lams2
# out = '   '
# for lam in lams:
#     out += ' & ' + str(lam)
# out += ' \\\\ \\hline\n'
# for M in [0, 1, 2, 3, 4, 5, 6, 7]:
#     out += '  ' + str(M)
#     for lam in lams:
#         cur = ridge(M, lam, X, Y)
#         out += ' & ' + '%.2f'%(cur)
#     out += ' \\\\\n'
# print out

M = 4
lam = 10
X, Y = regressAData()
Xv, Yv = regressBData()
fn = poly_fn(M)
Z = np.array([fn(x) for x in X])
weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(Z), Z), lam * np.identity(Z.shape[1]))), np.transpose(Z)), Y)
print sse(Xv, Yv, fn, weights)
