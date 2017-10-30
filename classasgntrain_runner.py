# file classasgntrain_runner.py

import neuralNetwork as NN
import matplotlib.pyplot as plt
import random
import numpy as np

# X and Y don't refer to coordinates in a
# plane. They refer to the inputs and the
# outputs.
X = []
Y = []

with open('classasgntrain1.dat', 'r') as fin:
	for line in fin:
		line = line.split()
		print(line)
		X.append([float(line[0]), float(line[1])])
		Y.append(0)
		X.append([float(line[2]), float(line[3])])
		Y.append(1)


# This plots the data
'''
for i in range(len(X)):
	pltMkr = ""
	if Y[i] == 0: # Class 0
		pltMkr = 'ro'
	elif Y[i] == 1: # Class 1
		pltMkr = 'go'
	plt.plot(X[i][0], X[i][1], pltMkr)
plt.show()
'''

############################################
# Split up the values into training data
# and testing data
############################################
num_4_test = len(X) // 5

X_test = []
Y_test = []
for i in range(num_4_test):
	idx = random.randint(0, len(X)) - 1
	X_test.append(X.pop(idx))
	Y_test.append(Y.pop(idx))

X_train = X
Y_train = Y

# Convert listy stuff to matrices
'''
X_test = np.matrix(X_test)
Y_test = np.matrix(Y_test)
X_train = np.matrix(X_train)
Y_train = np.matrix(Y_train)
'''

epochs = 100
num_neurons = [5, # Hidden layer neurons
			   1] # Output layer neurons

nn = NN.NeuralNetwork(num_neu = num_neurons, act_func = 'sigmoid')

for epoch in range(epochs):

	########################################
	# Test how well the neural network does
	########################################

	print('X_test', X_test)
	yhat = nn.forward(X_test)
	print('yhat', yhat)

	acc = 0
	for i in range(len(Y_test)):
		if (yhat[i] < 0.5 and Y_test[i] == 0) or (yhat[i] > 0.5 and Y_test[i] == 1):
			acc += 1
	acc = acc / len(Y_test)

	print(acc, 'correct.')



	#######################################
	# Do training via gradient descent
	# for all the training data
	#######################################
	for i in range(len(X_train)):
		yhat = nn.forward(X_train[i])
		#print('Y_train[i]', np.matrix(Y_train[i]))
		#print('yhat', yhat)
		#assert(False)
		nn.backward(np.matrix(Y_train[i]), yhat)
		




