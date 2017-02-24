import numpy as np 
import random
from keras.utils import np_utils

weight_init_scale = 0.01
# fix random seed for reproducibility
np.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print seq_in, '->', seq_out



hidden_size = 10
time_steps = 1
n_inputs = seq_length
n_outputs = 1
epochs = 100000

epsilon = 0.0001


# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), time_steps, seq_length))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
Y = np.array([[x/float(len(alphabet))] for x in dataY])




print X.shape
print Y.shape

print X[:3]
print Y[:3]

# w_hh = weight_init_scale * np.random.randn(hidden_size, hidden_size)
# 	# print w_hh.shape

# w_xh = weight_init_scale * np.random.randn(n_inputs, hidden_size)
# # print w_xh.shape

# w_hy = weight_init_scale * np.random.randn(hidden_size, n_outputs)
# # print w_hy.shape

w_hh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
	# print w_hh.shape

w_xh = np.random.randn(n_inputs, hidden_size) / np.sqrt(hidden_size)
# print w_xh.shape

w_hy = np.random.randn(hidden_size, n_outputs) / np.sqrt(hidden_size)
# print w_hy.shape


def get_random(data_x, data_y):
	index = random.randint(0, len(data_x) - 1)
	return data_x[index], data_y[index]

def ReLU_derivative(z):
	result = z[0]
	result[np.where(x > 0)] = 1.0
	result[np.where(x <= 0)] = 0.0

	return result

h_prev = np.zeros([1, hidden_size])

for epoch in range(epochs):

	x, y = get_random(X[:5], Y)

	

	h = np.tanh(h_prev.dot(w_hh.T) + x.dot(w_xh))
	# Relu
	# z = h_prev.dot(w_hh.T) + x.dot(w_xh)
	# h = np.maximum(z, 0.0)
	# print h.shape

	prediction = h.dot(w_hy)
	# print prediction.shape

	total_error = 0.5 * np.square(y - prediction)
	print "Epoch number: " + str(epoch) + "    total_error: " + str(total_error)

  	delta_error = (y - prediction)
  	# print delta_error.shape
  	delta_w_hy = epsilon * h.T.dot(delta_error)

  	# print delta_w_hy.shape
  	# print w_hy.shape 

  	delta_h = (1 - np.square(h)) * delta_error
  	# delta_h relu
  	# print z
  	# delta_h = ReLU_derivative(z) * delta_error
  	# print ReLU_derivative(z)

  	# print h_prev.shape
  	# print delta_h.shape


  	delta_hh = epsilon * h_prev.T.dot(delta_h)
  	# print delta_hh.shape

  	delta_xh = epsilon * x.T.dot(delta_h)
  	# print delta_xh.shape


  	# update weights

  	w_hy += delta_w_hy
  	w_hh += delta_hh
  	w_xh += delta_xh


 	h_prev = h




