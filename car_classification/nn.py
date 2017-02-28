import numpy as np 
import pandas as pd 


# Preprocessing

ds = pd.read_csv('car.data', sep=",", header=None)
ds.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']

ds['buying'].replace('vhigh', 1, inplace=True)
ds['buying'].replace('high', 2, inplace=True)
ds['buying'].replace('med', 3, inplace=True)
ds['buying'].replace('low', 4, inplace=True)


ds['maint'].replace('vhigh', 1, inplace=True)
ds['maint'].replace('high', 2, inplace=True)
ds['maint'].replace('med', 3, inplace=True)
ds['maint'].replace('low', 4, inplace=True)

ds['doors'].replace('2', 1, inplace=True)
ds['doors'].replace('3', 2, inplace=True)
ds['doors'].replace('4', 3, inplace=True)
ds['doors'].replace('5more', 4, inplace=True)

ds['persons'].replace('2', 1, inplace=True)
ds['persons'].replace('4', 2, inplace=True)
ds['persons'].replace('more', 3, inplace=True)

ds['lug_boot'].replace('small', 1, inplace=True)
ds['lug_boot'].replace('med', 2, inplace=True)
ds['lug_boot'].replace('big', 3, inplace=True)

ds['safety'].replace('low', 1, inplace=True)
ds['safety'].replace('med', 2, inplace=True)
ds['safety'].replace('high', 3, inplace=True)

ds['Class'].replace('unacc', 1, inplace=True)
ds['Class'].replace('acc', 2, inplace=True)
ds['Class'].replace('good', 3, inplace=True)
ds['Class'].replace('vgood', 4, inplace=True)

ds = ds[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']]



# Normalization

def normalize(column):
	return (column - column.mean()) / (column.max() - column.min())

buying = normalize(ds['buying'])
maint = normalize(ds['maint'])
doors = normalize(ds['doors'])
persons = normalize(ds['persons'])
lug_boot = normalize(ds['lug_boot'])
safety = normalize(ds['safety'])
Class = normalize(ds['Class'])



X = np.asarray(pd.concat([buying, maint, doors, persons, lug_boot, safety], axis=1, join='inner'))
Y = np.asarray(pd.get_dummies(ds['Class']))

print X[:10]
print Y[:10]

print X.shape
print Y.shape


################################################################################

# We use a 2 layers neural network with one 8 neurons hidden layer

n_inputs = 6
n_h1 = 25
n_outputs = 4
n_samples = len(X)
epochs = 100000
learning_rate = 1e-0
learning_rate = 0.01
reg=1e-3


def softmax(x):
	return np.array([np.exp(x[i])/(np.sum(np.exp(x), axis=1)[i]) for i in range(len(x))])

def sigmoid(x):
	return 1/(1+np.exp(-x))

def measure_accuracy(prediction, target):
	accuracy = 0
	for i in range(len(target)):
		if np.argmax(target[i]) == np.argmax(prediction[i]):
			accuracy += 1
	return float(accuracy)/len(target)




class Dense:

	def __init__(self, number_of_neurons, input_dim):
		self.number_of_neurons = number_of_neurons
		self.input_dim = input_dim
		self.weights = 0.01 * np.random.randn(input_dim, number_of_neurons)

	def forward(self, X):
		self.input = X
		self.output = X.dot(self.weights)
		return self.output

	def backward(self, dout):
		dW = self.input.T.dot(dout)
		dX = dout.dot(self.weights.T)

		return dX, dW

	def change_weights(self, delta):
		self.weights -= learning_rate * delta



class Activation:


	def __init__(self, function):
		self.function = function

	def forward(self, X):
		self.input = X

		if self.function == "ReLU":
			self.output = np.maximum(0, X)

		elif self.function == "tanh":
			self.output = np.tanh(X)

		elif self.function == "softmax":
			self.output = np.array([np.exp(X[i])/(np.sum(np.exp(X), axis=1)[i]) for i in range(len(X))])

		else:
			self.output = np.maximum(0, X)

		return self.output


	def backward(self, dout):
		 
		if self.function == "ReLU":
			gradient = np.empty_like(self.input)
			gradient[self.input <= 0] = 0
			gradient[self.input > 0] = 1

		elif self.function == "tanh":
			gradient = 1 - self.output**2

		elif self.function == "softmax":
			# SM = self.output.reshape((-1, 1))
			gradient = Y * self.output - self.output * self.output
			# gradient = Y * self.output - (np.reshape(np.sum(Y * self.output, axis=1),(-1, 1))*self.output)
			# gradient = 1
			# print "shape is " + str(np.reshape(np.diag(self.output), (-1, 1)).shape)
			# print "shape2 is " + str((self.output*self.output).shape)
			# print "shape2 is " + str((np.sum(self.output*self.output, axis=1)).shape)
			# gradient = ((dout - np.reshape(np.sum(dout*self.output, axis=1), (-1, 1))) * self.output)
			return gradient
		else:
			gradient = np.empty_like(self.input)
			gradient[self.input <= 0] = 0
			gradient[self.input > 0] = 1

		dX = dout * gradient

		return dX
		

# layer_1 = Dense(n_h1, n_inputs)
# layer_2 = Activation("softmax")
# z = layer_1.forward(X)
# print z.shape
# pred = layer_2.forward(z)
# print Y.shape








# w_l1 = 0.01 * np.random.randn(n_inputs, n_h1)
# w_output = 0.01 * np.random.randn(n_h1, n_outputs)
layer_1 = Dense(n_h1, n_inputs)
act_1 = Activation("ReLU")

layer_2 = Dense(n_outputs, n_h1)
act_2 = Activation("softmax")


for epoch in range(epochs):

	z1 = layer_1.forward(X)
	print z1.shape
	h1 = act_1.forward(z1)
	print h1.shape

	z2 = layer_2.forward(h1)
	prediction = act_2.forward(z2)

	# reg_loss = 0.5 * reg * np.sum(w_output*w_output) + 0.5 * reg * np.sum(w_l1 * w_l1)
	# loss =  - np.sum(np.sum(Y * np.log(prediction), axis=1), axis=0)/n_samples
	loss =  np.sum(-np.sum(Y * np.log(prediction), axis=1), axis=0)/n_samples

	# loss = np.sum(-np.log((Y * prediction)))/n_samples

	accuracy = measure_accuracy(prediction, Y)

	print "Epoch number: " + str(epoch) + "    loss: " + str(loss) + "    accuracy: " + str(accuracy)

	# output_delta = (prediction - Y)
	output_delta = -Y * np.reshape(1/np.sum((Y*prediction), axis=1), (-1, 1))

	# Sigmoid backprop
	# l1_delta = output_delta.dot(w_output.T) * h1 * (1 - h1)

	# tanh backprop
	# l1_delta = output_delta.dot(w_output.T) * (1 - h1**2)

	# ReLU backprop
	# l1_delta = output_delta.dot(w_output.T)
	# l1_delta[ h1 <= 0 ] = 0

	dout_act_2 = act_2.backward(output_delta)
	dout_layer_2, dw_layer_2 = layer_2.backward(dout_act_2)

	dout_act_1 = act_1.backward(dout_layer_2)
	dout_layer_1, dw_layer_1 = layer_1.backward(dout_act_1)


	layer_2.change_weights(dw_layer_2)
	layer_1.change_weights(dw_layer_1)
