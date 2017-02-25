from sklearn.datasets import load_digits
import numpy as np 
from im2col import im2col_indices, col2im_indices

filter_size = 3
n_filters = 20
n_filters_3 = 10
weights_scale = 0.01
epsilon = 0.00005
epochs = 1000


n_classes = 10

def set_item(array, index, value):
	array[index] = value
	return array

def get_one_encoded(Y):
	return np.array([ set_item(np.zeros(n_classes), y, 1) for y in Y ])


digits = load_digits()
print digits.data[0]
data = digits.data - np.mean(digits.data, axis=1, keepdims=True)
print data[0]
# print digits.data.shape

Y = digits.target
Y = get_one_encoded(Y)

data = np.reshape(data, (1797, 1, 8, 8))
# print data.shape

# import matplotlib.pyplot as plt 
# plt.gray() 
# plt.matshow(data) 
# plt.show()

def softmax(x):
	return np.array([np.exp(x[i])/(np.sum(np.exp(x), axis=1)[i]) for i in range(len(x))])

def conv_forward(X, W, stride=1, padding=1):
    cache = W, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    # if not h_out.is_integer() or not w_out.is_integer():
    #     raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col.dot(X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, stride, padding, X_col)

    return out, cache


def conv_backward(dout, cache):
    X, W, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped.dot(X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T.dot(dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW

def max_pool_forward(X, d, h_in, w_in, padding=0, stride=2):

	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	size = 2
	n = len(data)
	# d = n_filters

	h_out = (w_in - size)/stride + 1
	w_out = (h_in - size)/stride + 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	# print (len(data)*d, 1, h_in, w_in)
	X_reshaped = X.reshape(len(data)*d, 1, h_in, w_in)

	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

	# Next, at each possible patch location, i.e. at each column, we're taking the max index
	max_idx = np.argmax(X_col, axis=0)

	cache = (X, X_col, max_idx)

	# Finally, we get all the max value at each column
	# The result will be 1x9800
	out = X_col[max_idx, range(max_idx.size)]

	# Reshape to the output size: 14x14x5x10
	out = out.reshape(h_out, w_out, n, d)

	# Transpose to get 5x10x14x14 output
	out = out.transpose(2, 3, 0, 1)

	return out, cache

def max_pool_backward(dout, d, h_in, w_in, X, X_col, max_idx, stride=2):
	size = 2
	n = len(data)

	# X_col and max_idx are the intermediate variables from the forward propagation step

	# Suppose our output from forward propagation step is 5x10x14x14
	# We want to upscale that back to 5x10x28x28, as in the forward step

	# 4x9800, as in the forward step
	dX_col = np.zeros_like(X_col)

	# 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
	# Transpose step is necessary to get the correct arrangement
	dout_flat = dout.transpose(2, 3, 0, 1).ravel()

	# Fill the maximum index of each column with the gradient

	# Essentially putting each of the 9800 grads
	# to one of the 4 row in 9800 locations, one at each column
	dX_col[max_idx, range(max_idx.size)] = dout_flat

	# We now have the stretched matrix of 4x9800, then undo it with col2im operation
	# dX would be 50x1x28x28
	dX = col2im_indices(dX_col, (n * d, 1, h_in, w_in), size, size, padding=0, stride=stride)

	# Reshape back to match the input dimension: 5x10x28x28
	dX = dX.reshape(X.shape)

	return dX

def measure_accuracy(prediction, target):
	accuracy = 0
	for i in range(len(target)):
		if np.argmax(target[i]) == np.argmax(prediction[i]):
			accuracy += 1
	return float(accuracy)/len(target)


# X = im2col_indices(data, 3, 3)
# filters_1 = np.random.randn(n_filters, filter_size*filter_size)
filters_1 = np.random.randn(n_filters, 1, filter_size, filter_size) / np.sqrt((8*8*n_filters)/2)
filters_3 = np.random.randn(n_filters_3, n_filters, filter_size, filter_size) / np.sqrt((4*4*n_filters_3)/2)
fc_weights = np.random.randn(2*2*n_filters_3, n_classes) / np.sqrt((2*2*n_filters_3))
n_samples = len(data)

# print X.shape
# print filters_1.shape
# result = filters_1.dot(X)
# result = result.reshape(n_filters, 8, 8, len(data))
# result = result.transpose(3, 0, 1, 2)
# print result.shape

for epoch in range(epochs):
	out1, cache1 = conv_forward(data, filters_1)
	# print out1.shape

	out2, cache2 = max_pool_forward(out1, n_filters, 8, 8)
	# print out2.shape

	out3, cache3 = conv_forward(out2, filters_3)
	# print out3.shape

	out4, cache4 = max_pool_forward(out3, n_filters_3, 4, 4)
	# print out4.shape

	fc_input = out4.reshape(len(data), -1)
	

	output = fc_input.dot(fc_weights)
	prediction = softmax(output)

	loss =  np.sum(-np.sum(Y * np.log(prediction), axis=1), axis=0)/n_samples

	
	if epoch % 10 == 0:
		accuracy = measure_accuracy(prediction, Y)
		print "Epoch number: " + str(epoch) + "    loss: " + str(loss) + "    accuracy: " + str(accuracy)
	else:
		print "Epoch number: " + str(epoch) + "    loss: " + str(loss)



	# accuracy = measure_accuracy(prediction, Y)
	# print "Epoch number: " + str(epoch) + "    loss: " + str(loss) + "    accuracy: " + str(accuracy)

	

	# print prediction.shape
	# print prediction[0]

	# print Y.shape

	output_delta = prediction - Y
	# print output_delta.shape
	fc_weights_delta = fc_input.T.dot(output_delta)

	input_delta = output_delta.dot(fc_weights.T)
	# print input_delta.shape

	d_out4 = input_delta.reshape(len(data), n_filters_3, 2, 2)
	d_out3 = max_pool_backward(d_out4, n_filters_3, 4, 4, cache4[0], cache4[1], cache4[2])
	d_out2, d_filters_3 = conv_backward(d_out3, cache3)
	d_out1 = max_pool_backward(d_out2, n_filters, 8, 8, cache2[0], cache2[1], cache2[2])
	d_data, d_filters = conv_backward(d_out1, cache1)
	# print d_out2.shape

	fc_weights -= epsilon * fc_weights_delta 
	filters_1 -= epsilon * d_filters
	filters_3 -= epsilon * d_filters_3





