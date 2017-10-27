from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import numpy as np
import random
mnist = input_data.read_data_sets("/tmp/data/") # or wherever you want

# to put your data
X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")
X_test = mnist.test.images
y_test = mnist.test.labels.astype("int")
print("X_train.shape=",X_train.shape," y_train.shape=",y_train.shape)
print("X_test.shape=",X_test.shape," y_test.shape=",y_test.shape)

# plot one of these
print("X_train len:", len(X_train))

class NeuralNetwork:
	def __init__(self, num_neu=[0,0], act_func="softmax"):
		self.num_neu = num_neu
		self.has_run = False
		# TODO: add all the prime functions
		if act_func is "softmax":
			self.act_func = self.softmax
			self.deact_func = self.softmaxprime
			self.cost_func = self.softmax_cost
		elif act_func is "etothex":
			self.act_func = self.etothex
			self.deact_func = self.etothex # its own derivative
		elif act_func is "sigmoid":
			self.act_func = self.sigmoid

	def forward(self, x_in):
		self.assertValsSet()

		# make matrices that depend on the length of x_in
		self.A0 = np.matrix(x_in)
		batch_siz = len(x_in)
		self.A1 = np.zeros((batch_siz, self.num_neu[0]))
		self.A2 = np.zeros((batch_siz, self.num_neu[1]))
		self.errMat = np.zeros(self.A2.shape)	

		# Create weight matrices if not already created. The matrix
		# gynmastics here make the matrices contain random values 
		# between -1 and 1.
		if not self.has_run:
			self.has_run = True
			# 0th weight matrix, vals are (-1,1)
			self.W0 = np.random.rand(len(x_in[0]), self.num_neu[0])
			self.W0 -= (np.ones(self.W0.shape) / 2)
			self.W0 *= 2
				
		
			# a0 * W0 is a batch_siz x num_neu matrix	
			self.W1 = np.random.rand(self.num_neu[0], self.num_neu[1])
			self.W1 -= (np.ones(self.W1.shape) / 2)
			self.W1 *= 2

			if len(self.num_neu) > 2:
				# a0 * W0 is a batch_siz x num_neu matrix	
				self.W2 = np.random.rand(self.num_neu[1], self.num_neu[2])
				self.W2 -= (np.ones(self.W2.shape) / 2)
				self.W2 *= 2

		
		
		# From inpput to hidden layer
		self.Z1 = self.A0 * self.W0
		self.A1 = self.ReLU(self.Z1)

		self.Z2 = self.A1 * self.W1

		if len(self.num_neu) is 2:
			# From hidden layer to output
			self.A2 = self.act_func(self.Z2) #gives yhat

			return self.A2 #gives yhat
		else:
			# From 1st hidden layer to 2nd hidden layer
			self.A2 = self.ReLU(self.Z2)

			# From 2nd hidden layer to output
			self.Z3 = self.A2 * self.W2
			self.A3 = self.act_func(self.Z3) #gives yhat

			return self.A3
		

	def ReLU(self, mat):
		newmat = mat
		newmat[newmat<0] = 0
		return newmat

	def ReLUprime(self, mat):
		newmat = mat
		newmat[newmat < 0] = 0
		newmat[newmat > 0] = 1
		#newmat[newmat != None] = 1
		return newmat

	
	def softmax(self, mat):
		rows, cols = mat.shape
		yhat = np.zeros(self.A2.shape)
		for m in range(rows):
			# Find the alpha
			alpha = None
			for n in range(cols):
				alpha = mat[m,n] if alpha is None else max(alpha, mat[m,n])

			
			# Get all the e^zs for the row
			exponents = []
			for n in range(cols):
				exponents.append(np.exp(mat[m,n] - alpha))
			assert(len(exponents) == cols)

			denom = sum(exponents)
			assert(denom != 0)
			
			# find all the numerators and put them in
			for n in range(cols):
				yhat[m,n] = exponents[n] / denom
		return yhat

	def softmaxprime(self, yhat, y):
		# return np.multiply(yhat, (np.ones(yhat.shape) - yhat))
		return yhat - y

	def softmax_cost(self,yhat, y):
		assert(yhat.shape == y.shape)
		rows, cols = yhat.shape
		newMat = np.zeros((rows, 1))
		for m in range(rows):
			vals = []
			for n in range(cols):
				vals.append(y[m,n] * math.log(yhat[m,n] + .01))
			newMat[m,0] = -sum(vals)
		return newMat
	

	# TODO: finish this. contents are the softmax
	def sigmoid(self, mat):
		'''
		rows, cols = mat.shape
		yhat = np.zeros(self.A2.shape)
		for m in range(rows):
			for n in range(cols):
				yhat[m,n] = 1 + np.exp(-1 * mat[m,n]
				yhat[m,n] = 1 / yhat[m,n]
		return yhat
		'''
		pass

	def etothex(self, mat):
		rows, cols = mat.shape
		yhat = np.zeros(mat.shape)
		for m in range(rows):
			# Normalize
			alpha = None
			for n in range(cols):
				alpha = mat[m,n] if alpha is None else max(alpha, mat[m,n])
			
			for n in range(cols):
				yhat[m,n] = np.exp(mat[m,n] - alpha)
		return yhat
				

	def assertValsSet(self):
		assert(self.num_neu is not None)
	
	def backward(self, y_exp, yhat):

		scalar = 0.05

		# Get the changes needed for each weight matrix
		if len(num_neu) is 2:
			delta3 = yhat - y
			djdw1 = np.dot(self.A1.T, delta3)

			delta2 = np.multiply(np.dot(delta3, self.W1.T), self.ReLUprime(self.Z1))
			djdw0 = np.dot(self.A0.T, delta2)


		elif len(num_neu) is 3:
			delta4 = yhat - y
			djdw2 = np.dot(self.A2.T, delta4)

			delta3 = np.multiply(np.dot(delta4, self.W2.T), self.ReLUprime(self.Z2))
			djdw1 = np.dot(self.A1.T, delta3)

			delta2 = np.multiply(np.dot(delta3, self.W1.T), self.ReLUprime(self.Z1))
			djdw0 = np.dot(self.A0.T, delta2)

		self.W0 = self.W0 - scalar * djdw0 / np.max(np.fabs(djdw0))
		self.W1 = self.W1 - scalar * djdw1 / np.max(np.fabs(djdw1))
		if len(num_neu) is 3:
			self.W2 = self.W2 - scalar * djdw2 / np.max(np.fabs(djdw2))
	
	def dispWeights(self):
		print("W0:")
		print(self.W0)
		print("W1:")
		print(self.W1)

	def dispW0List(self):
		print("W0:")
		print(self.W0.tolist())
		


batch_siz = 50
num_neu_h1 = 300
num_neu_h2 = 10
num_outs = 10
nn = NeuralNetwork(num_neu=[num_neu_h1, num_outs], act_func="softmax")


# The main loop.
for tour in range(20):
	######################################################
	# This is where the testing begins. We just throw all
	# the test data at it at once in one big matrix using
	# the function forward().
	######################################################
	
	# construct the batch
	x_test_in = X_test
	y = np.zeros((x_test_in.shape[0], num_outs))
	for n in range(x_test_in.shape[0]):
		y[n,y_test[n]] = 1


	yhat = nn.forward(x_test_in)

	# Change yhat from matrix output to a list of values
	# just like how the y_test is formatted. Put yhat 
	# into outage list.
	outage = []
	rows, cols = yhat.shape
	for r in range(rows):
		max_idx = -1
		max_val = -1
		for c in range(cols):
			if yhat[r,c] > max_val:
				max_val = yhat[r,c]
				max_idx = c
		outage.append(max_idx)

	# Compare outage to y_test, get percent accuracy
	num_right = 0
	for idx in range(len(outage)):
		#print(y_test[idx], outage[idx])
		if y_test[idx] == outage[idx]:
			num_right += 1
	acc = num_right / len(outage)
	
	
	print(acc, "correct.")

	# Plot how good the accuracy is for this tour
	plt.plot(tour, acc * 100, "ro")

	##########################################################
	# This is where the training begins. We use forward() as
	# before, but we also use backward() to change the weights
	# and train the neural network.
	##########################################################

	for iters in range(100): # train it for some iterations

		# Get random indicies from X_train so we can have a new random batch.
		# we call the new random batch x_in
		idxs = [ random.randint(0, len(X_train) - 1) for i in range(batch_siz) ]
		x_in = [ X_train[n] for n in idxs ]

		# setup corresponding expected output from y_train and call it y
		y = np.zeros((batch_siz, num_outs))
		for n in range(len(idxs)):
			y[n,y_train[idxs[n]]] = 1
	
		yhat = nn.forward(x_in)

		# Apply changes to the weights via gradient descent
		nn.backward(y, yhat)



plt.show()


	
