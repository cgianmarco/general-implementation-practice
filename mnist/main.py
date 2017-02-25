from sklearn.datasets import load_digits
import numpy as np 
from im2col import im2col_indices, col2im_indices

filter_size = 3
n_filters = 20
n_filters_3 = 10

digits = load_digits()
print digits.data.shape

data = np.reshape(digits.data, (1797, 1, 8, 8))
print data.shape

# import matplotlib.pyplot as plt 
# plt.gray() 
# plt.matshow(data) 
# plt.show()

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

def max_pool_forward(X, d,  padding=0, stride=2):

	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	size = 2
	n = len(data)
	# d = n_filters

	h_out = (len(X) - filter_size)/stride + 1
	w_out = (len(X) - filter_size)/stride + 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	X_reshaped = X.reshape(len(data)*d, 1, 8, 8)

	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

	# Next, at each possible patch location, i.e. at each column, we're taking the max index
	max_idx = np.argmax(X_col, axis=0)

	# Finally, we get all the max value at each column
	# The result will be 1x9800
	out = X_col[max_idx, range(max_idx.size)]

	# Reshape to the output size: 14x14x5x10
	out = out.reshape(h_out, w_out, n, d)

	# Transpose to get 5x10x14x14 output
	out = out.transpose(2, 3, 0, 1)

	return out

# X = im2col_indices(data, 3, 3)
# filters_1 = np.random.randn(n_filters, filter_size*filter_size)
filters_1 = np.random.randn(n_filters, 1, filter_size, filter_size)
filters_3 = np.random.randn(n_filters_3, 1, filter_size, filter_size)
# print X.shape
# print filters_1.shape
# result = filters_1.dot(X)
# result = result.reshape(n_filters, 8, 8, len(data))
# result = result.transpose(3, 0, 1, 2)
# print result.shape

out1, cache1 = conv_forward(data, filters_1)
print out1.shape

out2 = max_pool_forward(out1, n_filters)
print out2.shape

out3, cache3 = conv_forward(out2, filters_3)
print out3.shape

out4 = max_pool_forward(out3, n_filters_3)



