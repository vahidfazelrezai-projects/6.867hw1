import pylab as pl
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def getGradientData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean, gaussCov, quadBowlA, quadBowlb) 

def getFittingData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y) 


def gradient_descent(objective_func, gradient_func, guess, step, threshold):
	prev = None
	delta = float("inf")
	mags = []
	while delta > threshold:
	# for i in range(2000):
		delta = abs(objective_func(guess) - objective_func(prev)) if prev is not None else float("inf")
		gradient = gradient_func(guess)
		mags.append(np.linalg.norm(gradient))
		prev = guess
		# mags.append(guess[0][0])
		guess = np.subtract(guess, step * gradient)
	# print "DONE", delta, threshold
	return guess, mags, delta

def stochastic_gradient_descent(objective_func, gradient_i, n, guess, threshold):
	prev = None
	delta = float("inf")
	mags = []
	r = 1
	n_t = lambda r : (r*1000 + 10000)**(-0.9)
	while delta > threshold:
		delta = abs(objective_func(guess) - objective_func(prev)) if prev is not None else float("inf")
		prev = guess
		for i in np.random.permutation(n):
			rate = n_t(r)
			gradient = gradient_i(guess, i)
			guess = np.subtract(guess, rate * gradient)
			r += 1
	return guess, mags, delta


def gaussian(mean, cov, delta):
	def gaussian(x):
		n = mean.shape[0]
		coeff = -1/math.sqrt(((2*np.pi)**n) * np.linalg.det(cov))
		a = np.transpose(np.subtract(x, mean)) 
		b = np.linalg.inv(cov)
		c = np.subtract(x, mean)
		return coeff * np.exp(-1./2 * a.dot(b).dot(c)[0][0])
	def gradient(x):
		return -gaussian(x) * np.linalg.inv(cov).dot(np.subtract(x, mean))
	def approx_gradient(x):
		delta_v = np.ones((len(x), 1))*0.5*delta
		return (gaussian(x + delta_v) - gaussian(x - delta_v))/delta/len(x)
	return gaussian, gradient, approx_gradient

def quadratic_bowl(A, B, delta):
	def quad(x):
		val = 0.5*x.T.dot(A).dot(x) - x.T.dot(B)
		return val
	def gradient(x):
		return A.dot(x) - B
	def approx_gradient(x):
		return (quad(np.add(x, delta)) - quad(x))
	return quad, gradient, approx_gradient

def least_squares_regression(X, y, delta):
	num_samples = y.shape[0]
	def lsq(theta):
		return np.linalg.norm(X.dot(theta) - y)**2
	def gradient(theta):
		return 2*X.T.dot(X.dot(theta) - y)
	def gradient_i(theta, i):
		x = np.array([X[i]])
		ret = 2*x.T*(x.dot(theta)[0][0] - y[i][0])
		return ret
	return lsq, gradient, gradient_i, num_samples

mean, cov, a, b = getGradientData()
mean = np.array([mean]).T
b = np.array([b]).T

# obj, grad, approx = gaussian(mean, cov, 0.00001)
# print "gaussian", obj(np.array([[8], [8]]))
# print "hello", grad(np.array([[8], [8]]))
# print ("approx", approx(np.array([[8], [8]])))

# # # all_mags = []
# steps = []
# thresholds = []
# errors = []
# obj, grad, approx = gaussian(np.array([[4.5], [3.4]]), np.array([[3, 4], [1, 2]]), 0.001)

# stepbounds = (0.0001, 0.001)
# # threshbounds = (0.0000000001, 0.0000000001*10)
# numsteps = 9
# # numthresh = 50
# for i in range(numsteps):
# 	thresh = 0.001
# 	step = stepbounds[0] + (stepbounds[1] - stepbounds[0])*1./numsteps*i
# 	guess, mags = gradient_descent(obj, grad, np.array([[8], [8]]), step, thresh)
# 	plt.plot(mags, label = str(step))

# plt.legend(title = "Step Size")
# plt.xlabel('Number of Iterations')
# plt.ylabel('Magnitude of Gradient')
# plt.show()
# # guessbounds = (0, 20)
# errors = []
# stepbounds = (1000, 10000)
# numsteps = 10
# # threshbounds = (0.00000000001, 0.00000000001*10)
# threshbounds = (0.0000000001, 0.0000000001*10)
# numthresh = 10
# for i in range(numsteps):
# 	step = stepbounds[0] + (stepbounds[1] - stepbounds[0])*1./numsteps*i
# 	for j in range(numthresh):
# 		thresh = threshbounds[0] + (threshbounds[1] - threshbounds[0])*1./numthresh*j
# 		# print thresh
# 		guess, mags, delta = gradient_descent(obj, grad, np.array([[8], [8]]), step, thresh)
# 		# plt.plot(mags)
# 		error = math.sqrt((guess[0][0] - 10)**2 + (guess[1][0] - 10)**2)
# 		print i, j, guess, delta
# 		errors.append(error)
# 		steps.append(step)
# 		thresholds.append(thresh)
# 		# print i, j
# plt.show()

# for g in range(10):
# 	thresh = 0.00000000001
# 	step = 1000
# 	guess, mags, delta = gradient_descent(obj, grad, np.array([[g], [g]]), step, thresh)
# 	plt.plot(mags, label = str(g))
# 	print mags[:10]
# 	error = math.sqrt((guess[0][0] - 10)**2 + (guess[1][0] - 10)**2)
# 	errors.append(error)
# 	print g, guess
# # steps.append(step)
# # thresholds.append(thresh)
# # print g, error
# #  print g
# # plt.plot(errors)
# # print [(steps[i], thresholds[i], errors[i]) for i in range(len(steps))]
# # plt.hexbin(steps, thresholds, C=errors, gridsize = 40, cmap=plt.cm.jet, bins=None)
# # plt.axis([min(steps), max(steps), min(thresholds), max(thresholds)])
# # print [abs(approx(np.array([[i], [i]])) - grad(np.array([[i], [i]]))) for i in range(10)]
# # cb = plt.colorbar()
# # cb.set_label('error')
# plt.legend(title="Initial Guess")
# plt.xlabel('Step Size')
# plt.ylabel('Magnitude of Gradient')
# plt.show()

delta = 0.00001
for i in range(12):
	obj, grad, approx = gaussian(mean, cov, delta)
	actual_val = grad(np.array([[8], [8]]))
	approx_val = approx(np.array([[8], [8]]))
	print delta, "&", approx_val, "\\\\"
	delta = delta * 10
# xbounds = (1, 10)
# numx  = 10
# deltabounds = (0.01, 0.001)
# numdelta = 100
# xs = []
# deltas = []
# errors = []

# for j in range(numdelta): 
# 	for i in range(numx):
# 		x = xbounds[0] + (xbounds[1] - xbounds[0])*1./numx*i
# 		delta = deltabounds[0] + (deltabounds[1] - deltabounds[0])*1./numdelta*j
# 		actual = grad(np.array([[x], [x]]))[0][0]
# 		app = approx(np.array([[x], [x]]))
# 		print "actual = ", actual, "approx = ", app
# 		error = actual - app
# 		xs.append(x)
# 		deltas.append(delta)
# 		errors.append(error)

# print([(xs[i], deltas[i], errors[i]) for i in range(len(errors))])
# plt.hexbin(xs, deltas, C=errors, gridsize = 30, cmap=plt.cm.jet, bins=None)
# plt.axis([min(xs), max(xs), min(deltas), max(deltas)])
# # print [abs(approx(np.array([[i], [i]])) - grad(np.array([[i], [i]]))) for i in range(10)]
# cb = plt.colorbar()
# cb.set_label('Least Squares Error')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Delta')
# plt.show()

# for i in range(10):
# 	s = 0.0001 + 0.0001*i
#  	guess, mags, = gradient_descent(obj, grad, np.array([[8], [8]]), s, 0.00000000001)
# 	plt.plot(mags, label = str(s))
# print [abs(approx(np.array([[i], [i]])) - grad(np.array([[i], [i]]))) for i in range(10)]
# plt.legend()
# plt.xlabel('Iterations')
# plt.ylabel('Magnitude of Gradient')
# plt.plot(mags)
# plt.show()
#
#
# X, y = getFittingData()
# y = np.array([y]).T
# X = np.array(X)
# print y.shape, X.shape, np.random.rand(10, 1).shape

# obj, grad, gradient_i, num_samples = least_squares_regression(X, y, 0)

# start = time.time()
# guess, mags, = gradient_descent(obj, grad, 100*np.random.rand(10, 1), 0.0000001, 0.001)
# x = time.time()
# print guess
# print x - start, "seconds"
# guess, mags = stochastic_gradient_descent(obj, gradient_i, num_samples, 100*np.random.rand(10, 1), 0.001)
# print guess
# print time.time() - x, "seconds"
# # plt.plot(mags)
# plt.show()
