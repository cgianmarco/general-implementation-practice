import numpy as np

n_inputs = 2
n_outputs = 1

epsilon = 0.0000001
epochs = 100



X = np.array([[n, 2*n] for n in range(100)])
print X.shape

W = np.random.randn(n_inputs, n_outputs)
print W.shape

Y = np.array([ [4 * n] for n in range(100)])
print Y.shape

prediction = np.dot(X, W)
print prediction.shape

error = 0.5 * np.square(Y - prediction)
print error.shape



for epoch in range(epochs):

	prediction = np.dot(X, W)

	error = 0.5 * np.square(Y - prediction)
	total_error = 0.5 * np.sum(np.square(Y - prediction))

	print "Epoch number: " + str(epoch) + "    total_error: " + str(total_error) + "\n"
	

	error_derivative = - np.sum(X*(Y - prediction))
	delta_w = - epsilon * error_derivative

	W = W + delta_w