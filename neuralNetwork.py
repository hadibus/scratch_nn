# file neuralNetwork.py

import math
import numpy as np

class NeuralNetwork:
	def __init__(self, num_neu=[0,0], act_func="softmax"):
		self.num_neu = num_neu
		self.has_run = False
		if act_func is "softmax":
			self.act_func = self.softmax
			self.deact_func = self.softmaxprime
			self.cost_func = self.softmax_cost
		elif act_func is "etothex":
			self.act_func = self.etothex
			self.deact_func = self.etothex # its own derivative
		elif act_func is "sigmoid":
			self.act_func = self.sigmoid
			self.deact_func = self.sigmoidprime

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
		
			self.Z3 = np.dot(self.A2, self.W2)
			self.A3 = self.act_func(self.Z3) #gives yhat

			assert(self.A3.shape == (batch_siz, self.num_neu[2]))
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
		yhat = np.zeros(mat.shape)
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
	
	def sigmoid(self, mat):
		return 1 / (1 + np.exp(-mat))

	def sigmoidprime(self, mat):
		return np.exp(-mat) / ((1 + np.exp(-mat)) **2)

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
		if len(self.num_neu) is 2:
			delta3 = yhat - y_exp
			djdw1 = np.dot(self.A1.T, delta3)

			delta2 = np.multiply(np.dot(delta3, self.W1.T), self.ReLUprime(self.Z1))
			djdw0 = np.dot(self.A0.T, delta2)


		elif len(self.num_neu) is 3:
			delta4 = yhat - y_exp
			djdw2 = np.dot(self.A2.T, delta4)

			delta3 = np.multiply(np.dot(delta4, self.W2.T), self.ReLUprime(self.Z2))
			djdw1 = np.dot(self.A1.T, delta3)

			delta2 = np.multiply(np.dot(delta3, self.W1.T), self.ReLUprime(self.Z1))
			djdw0 = np.dot(self.A0.T, delta2)

		self.W0 = self.W0 - scalar * djdw0 / np.max(np.fabs(djdw0))
		self.W1 = self.W1 - scalar * djdw1 / np.max(np.fabs(djdw1))

		if len(self.num_neu) is 3:
			self.W2 = self.W2 - scalar * djdw2 / np.max(np.fabs(djdw2))
	
