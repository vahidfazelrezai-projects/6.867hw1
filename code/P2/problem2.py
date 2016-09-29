import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

data = pl.loadtxt('P2/curvefittingp2.txt')
X = np.array(data[0,:])
Y = np.array(data[1,:])

def gradient_descent(objective_func, gradient_func, guess, step, threshold):
	prev = None
	delta = float("inf")
	while delta > threshold:
		delta = abs(objective_func(guess) - objective_func(prev)) if prev is not None else float("inf")
		gradient = gradient_func(guess)
		prev = guess
		guess = np.subtract(guess, step * gradient)
	return guess

# computes the ordinary least squares (OLS) weights
def regress(fn):
    Z = np.array([fn(x) for x in X])
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z), Z)), np.transpose(Z)), Y);

# plots given function in given range
def plot_curve(fn, lower_x, upper_x, width=2):
    x = [i/100.0 for i in range(100 * int(math.floor(lower_x)), 100 * int(math.ceil(upper_x)))]
    y = [fn(t) for t in x]
    plt.plot(x, y, lw=width)

# returns function that evaluates as a polynomial with given weights
def poly_fn(M):
    def poly(x):
        return np.array([(x ** i) for i in range(M+1)])
    return poly

# returns function that evaluates as a cos series with given weights
def cos_fn(M):
    def cos(x):
        return np.array([np.cos((i+1) * math.pi * x) for i in range(M+1)])
    return cos

# function used to generate the data points (before noise)
def generator(x):
    return math.cos(math.pi * x) + math.cos(2 * math.pi * x)

def add_poly_plot(M):
    fn = poly_fn(M)
    weights = regress(poly_fn(M))
    def curve(x):
        return np.dot(weights, fn(x))
    plot_curve(curve, min(X), max(X), 3)
    plt.title('Linear Regression (M=' + str(M) + ')' )

def add_fn_plot(weights, fn):
    def curve(x):
        return np.dot(weights, fn(x))
    plot_curve(curve, min(X), max(X), 3)

def make_plot():
    plt.plot(X, Y, 'o')
    plot_curve(generator, min(X), max(X), 1)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

def sse_objective(fn):
    def obj(weights):
        return np.linalg.norm(Y - np.dot(weights, fn(X))) ** 2
        # return sum([(Y[i] - np.dot(weights, fn(X[i]))) ** 2 for i in range(len(X))])
    return obj

def sse_gradient(fn):
    def grad(weights):
        # return -2 * np.dot(fn(X), (Y - np.dot(np.transpose(fn(X)), weights)))
        return 2*fn(X).dot(fn(X).T.dot(weights) - Y)
    return grad


def compare(fn, M, step, threshold):
    guess = [1] * (M+1)
    obj = sse_objective(fn)
    grad = sse_gradient(fn)

    weights1 = regress(fn)
    weights2 = gradient_descent(sse_objective(fn), sse_gradient(fn), guess, step, threshold)
    #weights3 = stochastic_gradient_descent(sse_objective(fn), sse_gradient(fn), guess, step, threshold)

    print weights1
    print weights2
    # print weights3

    add_fn_plot(weights1, fn)
    add_fn_plot(weights2, fn)
    # add_fn_plot(weights3, fn)

    make_plot()


##### 2.1
# add_poly_plot(5)
# make_plot()


##### 2.2
# see sse_objective and sse_gradient above

##### 2.3
# M = 3
# compare(poly_fn(M), M, .001, .000001)

##### 2.4
# M = 5
# compare(cos_fn(M), M, .001, .000001)
