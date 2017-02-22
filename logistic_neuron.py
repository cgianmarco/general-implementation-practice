import numpy as np

n_inputs = 2
n_outputs = 1

epsilon = 0.1
epochs = 100

def sigmoid(x):
	return 1/(1+np.exp(-x))

def derivative_of_sigmoid(y):
	return y*(1-y)

X = np.array([[float(n), float(2*n)] for n in range(100)])
print X.shape

W = np.random.randn(n_inputs, n_outputs)
print W.shape

Y = np.array([ [(n**2)/10000.0] for n in range(100)])
print Y.shape

prediction = sigmoid(np.dot(X, W))
print prediction.shape



for epoch in range(epochs):

	prediction = sigmoid(np.dot(X, W))

	error = 0.5 * np.square( Y - prediction )
	total_error = 0.5 * np.sum(np.square(Y - prediction))

	print "Epoch number: " + str(epoch) + "    total_error: " + str(total_error)


	# error_derivative = - np.sum(X * prediction * (1-prediction) * (Y - prediction))
	# error_derivative = - np.dot(np.transpose(X), prediction * (1-prediction) * (Y - prediction))
	delta_error = (Y - prediction)
	delta_z = prediction * (1-prediction) * delta_error
	# print error_derivative

	# delta_w = - epsilon * error_derivative
	delta_w = epsilon * X.T.dot(delta_z)

	W = W + delta_w


# TODO: find a better dataset
X_test = np.array([[80, 160]])
predicted = sigmoid(np.dot(X_test, W))
expected = (80**2)/10000.0

print "Predicted: " + str(predicted) + "   Expected: " + str(expected)



