import matplotlib.pyplot as plt
import pylab as pl

def plot_points(X, Y):
    plt.plot(X, Y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')

def regress(X, Y, M):
    return [-1, 1]

def run():
    # load data
    data = pl.loadtxt('curvefittingp2.txt')
    X = data[0,:]
    Y = data[1,:]

    # plot points
    plot_points(X, Y)

    weights = regress(X, Y, 2)
    regressed_Y = []
    for x in X:
        y = 0
        for i in range(len(weights)):
            y += weights[i] * (x ** i)
        regressed_Y.append(y)

    plt.plot(X, regressed_Y, lw=2)

    plt.show()

run()
