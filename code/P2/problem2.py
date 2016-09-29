import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# computes the ordinary least squares (OLS) weights
def regress(X, Y, M):
    mat = []
    for x in X:
        row = []
        for i in range(M+1):
            row.append(x ** i)
        mat.append(row)
    Z = np.array(mat)
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z), Z)), np.transpose(Z)), Y);

# plots given points
def plot_points(X, Y):
    plt.plot(X, Y, 'o')

# plots given function in given range
def plot_curve(fn, lower_x, upper_x, width=2):
    x = [i/100.0 for i in range(100 * int(math.floor(lower_x)), 100 * int(math.ceil(upper_x)))]
    y = []
    for t in x:
        y.append(fn(t))
    plt.plot(x, y, lw=width)

# returns function that evaluates as a polynomial with given weights
def poly_fn(weights):
    def poly(x):
        return sum([weights[i] * (x ** i) for i in range(len(weights))])
    return poly

# function used to generate the data points (before noise)
def generator(x):
    return math.cos(math.pi * x) + math.cos(2 * math.pi * x)

# makes plot appropriate for problem 2.1
def make_plot(M):
    # load data
    data = pl.loadtxt('P2/curvefittingp2.txt')
    X = np.array(data[0,:])
    Y = np.array(data[1,:])

    # plot points
    plot_points(X, Y)

    # compute and plot polynomial
    weights = regress(X, Y, M)
    plot_curve(poly_fn(weights), min(X), max(X), 3)

    # plot generator curve
    plot_curve(generator, min(X), max(X), 1)

    # touch up and display plot
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Linear Regression (M=' + str(M) + ')' )
    plt.show()
