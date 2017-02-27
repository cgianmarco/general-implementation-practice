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
n_h1 = 20
n_outputs = 4
n_samples = len(X)
epochs = 100000
learning_rate = 1e-0
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
		self.weights = 0.01 * np.random.randn(n_inputs, number_of_neurons)

	def forward(self, X):
		return X.dot(self.weights)

	def backward(selfs):
		pass



class Activation:

	def forward(self, X):
		return np.maximum(0, X)

	def backward(selfs):
		pass

		

layer_1 = Dense(n_h1, n_inputs)
layer_2 = Activation()
z = layer_1.forward(X)
print z.shape
print layer_2.forward(z).shape


# w_l1 = 0.01 * np.random.randn(n_inputs, n_h1)
# w_output = 0.01 * np.random.randn(n_h1, n_outputs)


# for epoch in range(epochs):

# 	h1 = np.maximum(0, X.dot(w_l1))
# 	assert h1.shape == (n_samples, n_h1), "Error on first layer"

# 	prediction = softmax(h1.dot(w_output))
# 	assert prediction.shape == (n_samples, n_outputs), "Error on second layer"


# 	reg_loss = 0.5 * reg * np.sum(w_output*w_output) + 0.5 * reg * np.sum(w_l1 * w_l1)
# 	loss =  np.sum(-np.sum(Y * np.log(prediction), axis=1), axis=0)/n_samples + reg_loss

# 	accuracy = measure_accuracy(prediction, Y)

# 	print "Epoch number: " + str(epoch) + "    loss: " + str(loss) + "    accuracy: " + str(accuracy)

# 	output_delta = (prediction - Y)

# 	# Sigmoid backprop
# 	# l1_delta = output_delta.dot(w_output.T) * h1 * (1 - h1)

# 	# tanh backprop
# 	# l1_delta = output_delta.dot(w_output.T) * (1 - h1**2)

# 	# ReLU backprop
# 	l1_delta = output_delta.dot(w_output.T)
# 	l1_delta[ h1 <= 0 ] = 0

# 	w_output_delta = h1.T.dot(output_delta) * reg
# 	w_l1_delta = X.T.dot(l1_delta) * reg


# 	w_output -= learning_rate * w_output_delta
# 	w_l1 -= learning_rate * w_l1_delta 
