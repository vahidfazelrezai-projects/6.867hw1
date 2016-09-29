import pylab as pl
import numpy as np
import math
import matplotlib.pyplot as plt

def getData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean, gaussCov, quadBowlA, quadBowlb) 


def gradient_descent(objective_func, gradient_func, guess, step, threshold, plot = False):
	prev = None
	delta = float("inf")
	mags = []
	while delta > threshold:
		delta = abs(objective_func(guess) - objective_func(prev)) if prev is not None else float("inf")
		gradient = gradient_func(guess)
		mags.append(np.linalg.norm(gradient))
		prev = guess
		guess = np.subtract(guess, step * gradient)
	return guess, mags

def gaussian(mean, cov, delta):
	def gaussian(x):
		n = mean.shape[0]
		coeff = -1/math.sqrt(((2*np.pi)**n) * np.linalg.det(cov))
		a = np.transpose(np.subtract(x, mean)) 
		b = np.linalg.inv(cov)
		c = np.subtract(x, mean)
		return coeff * np.exp(-1/2 * a.dot(b).dot(c)[0][0])
	def gradient(x):
		return -gaussian(x) * np.linalg.inv(cov).dot(np.subtract(x, mean))
	def approx_gradient(x):
		return gaussian(np.add(x, delta)) - gaussian(x)
	return gaussian, gradient, approx_gradient

def quadratic_bowl(A, B, delta):
	def quad(x):
		val = 0.5*x.T.dot(A).dot(x) - x.T.dot(B)
		return val
	def gradient(x):
		return A.dot(x) - B
	def approx_gradient(x):
		return quad(np.add(x, delta)) - quad(x)
	return quad, gradient, approx_gradient

mean, cov, a, b = getData()
mean = np.array([mean]).T
b = np.array([b]).T

obj, grad, approx = gaussian(mean, cov, np.array([[8], [8]]))
guess, mags = gradient_descent(obj, grad, np.array([[8], [8]]), 1000, 0.00000000001, True)
# print [abs(approx(np.array([[i], [i]])) - grad(np.array([[i], [i]]))) for i in range(10)]
plt.xlabel('Iterations')
plt.ylabel('Magnitude of Gradient')
plt.plot(mags)
plt.show()

obj, grad, approx = quadratic_bowl(a, b, np.array([[1], [1]]))
guess, mags, = gradient_descent(obj, grad, np.array([[8], [8]]), 0.0001, 0.00000000001, True)
# print [abs(approx(np.array([[i], [i]])) - grad(np.array([[i], [i]]))) for i in range(10)]
plt.xlabel('Iterations')
plt.ylabel('Magnitude of Gradient')
plt.plot(mags)
plt.show()

