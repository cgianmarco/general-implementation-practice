import numpy as np 

x = np.array([10.0, 9.0, 7.0, 5.0])
Y = np.array([1, 0, 0, 0])
epsilon = 1e-5

def my_func(x):
	return x**2

def my_grad(x):
	return 2*x

def softmax(x):
	return np.exp(x)/(np.sum(np.exp(x), axis=0))

def softmax_grad(x):
	return Y*softmax(x) - np.max(softmax(x)) * softmax(x) 

def log_loss(x):
	return -Y * np.log(x)

def log_loss_gradient(x):
	return - Y * 1/np.sum((Y*x), axis=0)



# def softmax(x):
# 	return np.exp(x)/(np.sum(np.exp(x), axis=0))

# # def softmax_grad(x):
# # 	return gradient = Y * self.output - self.output * self.output

# def log_loss(x):
# 	return -np.sum(Y * np.log(softmax(x)), axis=0)

# def log_loss_gradient(x):
# 	return -Y * 1/np.sum((Y*x), axis=0)

# print log_loss(x)
# print log_loss_gradient(x)


# def gradient_check(my_func, my_grad, margin):

# 	# calc_grad =	(my_func(x + Y*epsilon) - my_func(x))/(epsilon)
# 	calc_grad =	(my_func(x + Y*epsilon) - my_func(x - Y*epsilon))/(2*epsilon)
# 	# print "x - epsilon is " + str(my_func(x - epsilon))
# 	# print "x + epsilon is " + str(my_func(x + epsilon))
# 	print "calculated is " + str(calc_grad)
# 	return [ (my_grad(x)[i] <= calc_grad[i] + margin) and (my_grad(x)[i] >= calc_grad[i] - margin) for i in range(len(x))]


def gradient_check(my_func, my_grad, margin):

	# calc_grad =	(my_func(x + Y*epsilon) - my_func(x))/(epsilon)
	calc_grad =	(my_func(x + epsilon) - my_func(x - epsilon))/(2*epsilon)
	# print "x - epsilon is " + str(my_func(x - epsilon))
	# print "x + epsilon is " + str(my_func(x + epsilon))
	print "calculated is " + str(calc_grad)
	return [ (my_grad(x)[i] <= calc_grad[i] + margin) and (my_grad(x)[i] >= calc_grad[i] - margin) for i in range(len(x))]

print softmax(x)
print "my grad is " + str(log_loss_gradient(x))


print gradient_check(log_loss, log_loss_gradient, 1e-5)