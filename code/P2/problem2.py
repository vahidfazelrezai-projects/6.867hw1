import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def plot_points(X, Y):
    plt.plot(X, Y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')

def regress(X, Y, M):
    mat = []
    for x in X:
        row = []
        for i in range(M+1):
            row.append(x ** i)
        mat.append(row)
    Z = np.array(mat)
    # w = (X' X)^-1 X' y
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z), Z)), np.transpose(Z)), Y);

def curve_plot(fn, lower_x, upper_x, width=2):
    x = [i/100.0 for i in range(100 * int(math.floor(lower_x)), 100 * int(math.ceil(upper_x)))]
    y = []
    for t in x:
        y.append(fn(t))
    plt.plot(x, y, lw=width)

def run(M):
    # load data
    data = pl.loadtxt('P2/curvefittingp2.txt')
    X = np.array(data[0,:])
    Y = np.array(data[1,:])

    # plot points
    plot_points(X, Y)

    # compute and plot polynomial
    weights = regress(X, Y, M)
    def poly(x):
        return sum([weights[i] * (x ** i) for i in range(len(weights))])
    curve_plot(poly, min(X), max(X), 3)

    def generator(x):
        return math.cos(math.pi * x) + math.cos(2 * math.pi * x)
    curve_plot(generator, min(X), max(X), 1)

    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Linear Regression (M=' + str(M) + ')' )
    plt.show()
