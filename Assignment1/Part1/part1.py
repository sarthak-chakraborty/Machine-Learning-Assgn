import numpy as np 
import matplotlib.pyplot as plt
import math
import csv

# Number of iterations of the gradient descent to be performed.
n_iter = 5000


# Function that generates the sample and stores as a tuple
def genSample(sample):
	a = np.random.uniform(0,1,n_sample)
	b = np.sin(2*np.pi*np.array(a)).reshape(-1)
	b += np.random.normal(loc=0,scale=0.3,size=n_sample)
	sample.append([(a[i], b[i]) for i in range(len(a))])


# Function to split the sample into train and tet set
def splitSample(sample, train, test):
	tr = []
	ts = []
	for i in sample:
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
			err = np.transpose((h-y).dot(np.transpose(x)))	# np.sum() is not required since it is a vectorised implementation and performed using matrix multiplication
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
n_sample = 10
learning_rate = 0.05
deg = [i for i in range(1,10)]
sample= []
train = []
test = []
weights = [[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))]
err_train = [0 for i in range(len(deg))]
err_test = [0 for i in range(len(deg))]

# Generating the sample
print("Generating " + str(n_sample) + " samples...")
genSample(sample)
print("Dataset Generated")
sample = sample[0]

# Storing the generated data in a csv file
with open('dataset.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['X','Y'])
    for row in sample:
        csv_out.writerow(row)

# Split the generated dataset
splitSample(sample, train, test)
train = train[0]
test = test[0]
print("\nDataset split into 80% training data and 20% test data")

# Fits the curve and estimate the error in train and test set
print("\nFitting a curve of " + str(n_sample) + " samples using linear regression...")
fitCurveSquared(train, weights)
print("Curve fitting over")
estimateError(train, weights, err_train)
estimateError(test, weights, err_test)

print("\n\n")
for i in range(len(deg)):
	print("Degree of Polynomial n: " + str(deg[i]))
	print("Weights Learned: ")
	print(np.transpose(weights[i]))
	print("Training Error: " + str(err_train[i]))
	print("Test Error: " + str(err_test[i]))
	print("\n")

# Save the weights, and the error in an npy file to be used by part 2
np.save("dataset",sample)
np.save("weights", weights)
np.save("train error",err_train)
np.save("test error",err_test)

print("\n\n")
print("Over..")