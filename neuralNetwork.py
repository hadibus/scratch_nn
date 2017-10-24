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

class NeuralNetwork1:
	def __init__(self, num_neu=0, num_outs=0, act_func="softmax"):
		self.num_neu = num_neu
		self.num_outs = num_outs
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
		self.A1 = np.zeros((batch_siz, self.num_neu))
		self.A2 = np.zeros((batch_siz, self.num_outs))
		self.errMat = np.zeros(self.A2.shape)	

		# Create weight matrices if not already created.
		if not self.has_run:
			self.has_run = True
			# 0th weight matrix, vals are (-1,1)
			self.W0 = np.random.rand(len(x_in[0]), self.num_neu)
			self.W0 -= (np.ones(self.W0.shape) / 2)
			self.W0 *= 2
		
			# a0 * W0 is a batch_siz x num_neu matrix	
			self.W1 = np.random.rand(self.num_neu, self.num_outs)
			self.W1 -= (np.ones(self.W1.shape) / 2)
			self.W1 *= 2

		
		
		# From inpput to hidden layer
		self.Z1 = self.A0 * self.W0
		self.A1 = self.ReLU(self.Z1)

		# From hidden layer to output
		self.Z2 = self.A1 * self.W1
		self.A2 = self.act_func(self.Z2) #gives yhat
		return self.A2 #gives yhat
		

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
		assert(self.num_outs is not None)
	
	def backward(self, y_exp, yhat):
		
		#delta3 = np.multiply(-(y_exp - yhat), self.deact_func(self.Z2, self.A2))
		delta3 = yhat - y
		# delta3 = np.multiply(yhat - y_exp
		djdw1 = np.dot(self.A1.T, delta3)
		# print(djdw2.shape) # Esta bien!
	
		# djdw0 = djdw * self.W1.T * self.ReLUprime(self.Z1)
		delta2 = np.multiply(np.dot(delta3, self.W1.T), self.ReLUprime(self.Z1))
		djdw0 = np.dot(self.A0.T, delta2)
		
		# print(djdw0.shape)
		# print(self.W0.shape)

		scalar = 0.05
		self.W0 = self.W0 - scalar * djdw0 / np.max(np.fabs(djdw0))
		# max_val = np.max(self.W0)
		# min_abs_val = np.fabs(np.min(self.W0))
		# self.W0 /= max(max_val, min_abs_val)
		# max_val = np.max(djdw0)
		# min_abs_val = np.fabs(np.min(djdw0))
		#self.W0 = self.W0 - scalar * djdw0 / max(max_val, min_abs_val)


		self.W1 = self.W1 - scalar * djdw1 / np.max(np.fabs(djdw1))
		# max_val = np.max(self.W1)
		# max_val = np.max(np.fabs(self.W1))
		# self.W1 /= max(max_val, min_abs_val)
		# max_val = np.max(djdw1)
		# min_abs_val = np.fabs(np.min(djdw1))
		# self.W1 = self.W1 - scalar * djdw1 / max(max_val, min_abs_val)


		
	def loss(self, y, yhat):
		diff = y - yhat
		rows, cols = diff.shape
		newMat = np.multiply(diff,diff)
		addage = []
		for m in range(rows):
			rowSum = 0
			for n in range(cols):
				rowSum += newMat[m,n]
			addage.append(rowSum)
		# print(sum(addage) / rows)
		return 0.5 * sum(addage) / rows
	
	def dispWeights(self):
		print("W0:")
		print(self.W0)
		print("W1:")
		print(self.W1)

	def dispW0List(self):
		print("W0:")
		print(self.W0.tolist())
		


batch_siz = 50
num_neu = 300
num_outs = 10
nn = NeuralNetwork1(num_neu=num_neu, num_outs=num_outs, act_func="softmax")


# The main training loop.
for tour in range(20):
	
	# setup corresponding expected output.
	#test_batch_siz = 100
	x_test_in = X_test#[0:test_batch_siz]
	y = np.zeros((x_test_in.shape[0], num_outs))
	for n in range(x_test_in.shape[0]):
		y[n,y_test[n]] = 1


	yhat = nn.forward(x_test_in)

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

	num_wrong = 0
	for idx in range(len(outage)):
		#print(y_test[idx], outage[idx])
		if y_test[idx] != outage[idx]:
			num_wrong += 1
	acc = num_wrong / len(outage)
	
	print(acc, "incorrect.")
	plt.plot(tour, acc * 100, "ro")

	iter_nums = []
	errs = []
	for iters in range(100): # train it for some iterations
		# choose indices, make input.
		idxs = [ random.randint(0, len(X_train) - 1) for i in range(batch_siz) ]
		x_in = [ X_train[n] for n in idxs ]

		# setup corresponding expected output.
		y = np.zeros((batch_siz, num_outs))
		for n in range(len(idxs)):
			y[n,y_train[idxs[n]]] = 1
	

		yhat = nn.forward(x_in)

		costMat = nn.cost_func(yhat, y)
		cost = 0
		for m in range(costMat.shape[0]):
			cost += costMat[m,0]
		cost /= costMat.shape[0]
		iter_nums.append(iters)
		errs.append(cost)
		nn.backward(y, yhat)

	#print("Training iteration.")



plt.show()

#print(np.array(outage))	
#print(y_test[0:test_batch_siz])


#error = nn.loss(yhat)

	
