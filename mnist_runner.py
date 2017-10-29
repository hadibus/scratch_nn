from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random
import neuralNetwork as NN

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

tours = 4
batch_siz = 50
num_neurons = [300, 100, 10]
nn = NN.NeuralNetwork(num_neu=num_neurons, act_func="softmax")

plot_x = []
plot_y = []

# The main loop.
for tour in range(tours):
	######################################################
	# This is where the testing begins. We just throw all
	# the test data at it at once in one big matrix using
	# the function forward().
	######################################################
	
	# construct the batch
	x_test_in = X_test
	y = np.zeros((x_test_in.shape[0], num_neurons[-1]))
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
	plot_x.append(tour)
	plot_y.append(acc)

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
		y = np.zeros((batch_siz, num_neurons[-1]))
		for n in range(len(idxs)):
			y[n,y_train[idxs[n]]] = 1
	
		yhat = nn.forward(x_in)

		assert(y.shape == yhat.shape)

		# Apply changes to the weights via gradient descent
		nn.backward(y, yhat)


plt.plot(plot_y)
plt.title('MNIST training with ' + str(len(num_neurons) - 1) + ' hidden layers'  , fontsize=20)
plt.xlabel('Iteration')
plt.ylabel('Proportion correct')
plt.xticks(np.arange(min(plot_x), max(plot_x)+1, 1.0))
plt.show()


	
