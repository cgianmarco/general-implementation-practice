class ComputationalGraph:

	 def __init__(self, X, Y):
	 	self.layers = []
	 	self.input = X
	 	self.target = Y

	 def add(self, layer):
	 	self.layers.append(layer)

	 def setup(self):
	 	pass

	 def forward(self):

	 	result = self.input

	 	for layer in self.layers:

	 		result = layer.forward(result)

	 	self.prediction = result

	 	return result


	 def backward(self):
	 	dout = self.getLoss()
	 	
	 	for layer in reversed(self.layers):

	 		dout = layer.backward(dout)