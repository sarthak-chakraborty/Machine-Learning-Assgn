import numpy as np 
import matplotlib.pyplot as plt
import math

# Number of iterations of the gradient descent to be performed.
n_iter = 5000


# Function that generates the sample and stores as a tuple
def genSample(sample):
	for j in range(len(n_sample)):
		a = np.random.uniform(0,1,n_sample[j])
		b = np.sin(2*np.pi*np.array(a)).reshape(-1)
		b += np.random.normal(loc=0,scale=0.3,size=n_sample[j])
		sample.append([(a[i], b[i]) for i in range(len(a))])


# Function to split the sample into train and tet set
def splitSample(sample, train, test):
	for j in range(len(sample)):
		tr = []
		ts = []
		for i in sample[j]:
			n = np.random.uniform(0,1)
			if(n>0.8):
				ts.append(i)
			else:
				tr.append(i)
		train.append(tr)
		test.append(ts)


# Performs gradient descent on Squared Error Cost function
def fitCurveSquared(train, weights):
	for i in range(len(deg)):
		# Generate the feature vector and get the weight vector in proper format
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(train[k][0],j) for j in range(deg[i]+1)] for k in range(len(train))]

		x = np.array(x)
		x = np.transpose(x)
		weight = np.array(weight)

		# Update the weight for n_iter iterations
		for n in range(n_iter):
			h = np.transpose(weight).dot(x)
			y = np.array(zip(*train)[1])
			y = y.reshape(h.shape)
			err = np.transpose((h-y).dot(np.transpose(x)))
			weight -= (err * learning_rate)/len(train)
		weights[i] = weight


# Function that estimates the mean squared error given the weights and the data
def estimateError(data, weights, err):
	for i in range(len(deg)):
		# Initialize the weights and the feature vector in proper order
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(data[k][0],j) for j in range(deg[i]+1)] for k in range(len(data))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)

		# Calculate the squared error
		h = np.matmul(np.transpose(weight), x)
		y = np.array(zip(*data)[1])
		y = y.reshape(h.shape)
		a = (h - y).reshape(-1)*(h - y).reshape(-1)
		err[i] = np.sum(a)
		err[i] /= 2*len(x)



# Initialization of different values
n_sample = [10, 100, 1000, 10000]
learning_rate = 0.05
deg = [i for i in range(1,10)]
sample = []
train = []
test = []
weights = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(n_sample))]
err_train = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
err_test = [[0 for i in range(len(deg))] for k in range(len(n_sample))]

# Generate and split the sample intrain and test set. The samples are stored in a 2D matrix of shape (len(n_samples),n_samples)
print("Generating " + str(n_sample) + " samples...")
genSample(sample)
print("Dataset Generated")
splitSample(sample, train, test)
print("\nDataset split into 80% training data and 20% test data")

# For each sample size
for i in range(len(n_sample)):

	# Fit the regression curve and estimate the squared error
	print("\nFitting a curve of " + str(n_sample[i]) + " samples using linear regression...")
	fitCurveSquared(train[i], weights[i])
	print("Curve fitting over")
	estimateError(train[i], weights[i], err_train[i])
	estimateError(test[i], weights[i], err_test[i])
	print("")
	for j in range(len(deg)):
		print("Degree of Polynomial n: " + str(deg[j]))
		print("Weights Learned: ")
		print(np.transpose(weights[i][j]))
		print("Training Error: " + str(err_train[i][j]))
		print("Test Error: " + str(err_test[i][j]))
		print("")
	print("\n")

# Generates the regression curve and plots it
for i in range(len(n_sample)):
	x = [sample[i][k][0] for k in range(len(sample[i]))]
	y = [sample[i][k][1] for k in range(len(sample[i]))]
	for j in range(len(deg)):
		plt.figure()
		plt.scatter(x,y)
		x1 = np.linspace(0, 1, 100)
		y1 = []
		for k in x1:		# finds the value of the function using x values and the parameters of the function
			a = 0
			for l in range(deg[j]+1):
				a += weights[i][j][l]*math.pow(k,l)
			y1.append(a)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("REGRESSION LINE, M  = "+str(n_sample[i])+", DEGREE = "+str(deg[j]))
		plt.plot(x1, y1, "k")
		# plt.savefig("M = "+str(n_sample[i])+", Degree "+str(deg[j]))

	# Plots the train and test error vs Degree of polynomial
	plt.figure()
	plt.plot(deg, err_train[i], "b")
	plt.plot(deg, err_test[i], "r")
	plt.xlabel("Degree of Polynomial")
	plt.ylabel("Error Value")
	plt.title("ERROR vs DEGREE of Regression Line, M = "+str(n_sample[i]))
	plt.legend(["Train","Test"])
	plt.xticks(deg)
	# plt.savefig("Error, M "+str(n_sample[i]))


# Plot the Learning Curve for each degree of polynomial
for k in range(len(deg)):
	tr_err = zip(*err_train)[k]
	ts_err = zip(*err_test)[k]
	plt.figure()
	plt.plot(n_sample, tr_err, color="b")
	plt.plot(n_sample, ts_err, color="r")
	plt.xlabel("Number of Samples")
	plt.ylabel("Error Value")
	plt.title("LEARNING CURVE, DEGREE = "+str(deg[k]))
	plt.legend(["Train Error","Test Error"])
	# plt.savefig("Learning Curve, Degree "+str(deg[k]))
	plt.show()
print("\n\n")
print("Over..")