import numpy as np 
import random

n_inputs = 4
n_outputs = 2

epochs = 10
number_of_samples = 10
epsilon = 0.001

def random_sequence(n):
	return [random.randint(0,100)/1.0 for x in range(n)]

X = np.array([random_sequence(n_inputs) for x in range(number_of_samples)])
print X.shape

W = np.random.randn(n_inputs, n_outputs)
print W.shape

Y = np.array([random_sequence(n_outputs) for x in range(number_of_samples)])
print Y.shape

def sigmoid(x):
	return 1/(1+np.exp(-x))

def derivative_of_sigmoid(y):
	return y*(1-y)


prediction = sigmoid(np.dot(X, W))

print prediction


for epoch in range(epochs):

	prediction = sigmoid(np.dot(X, W))

	error = 0.5 * np.square( Y - prediction )
	total_error = 0.5 * np.sum(np.square(Y - prediction), axis=0)

	print total_error.shape

	print "Epoch number: " + str(epoch) + "    total_error: " + str(total_error)

	print (prediction * (1-prediction)).shape
	print ((Y - prediction)).shape
	print (X).shape

	#error_derivative = np.sum()
	error_derivative = - np.dot(np.transpose(X), prediction * (1-prediction) * (Y - prediction))
	print error_derivative.shape

	delta_w = - epsilon * error_derivative
	print delta_w.shape

	print W
	print delta_w
	W = W + delta_w
	print W
	print W.shape

# TODO: testing on real dataset

