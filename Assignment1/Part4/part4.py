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
def fitCurveSquared(train, weights, learning_rate):
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



# Performs gradient descent on Absolute Mean Error Cost function
def fitCurveAbs(train, weights, learning_rate):
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
			a = (h - y)/abs(h-y)	# To get the matrix with values either 1 or -1
			err = np.transpose(a.dot(np.transpose(x)))
			weight -= (err * learning_rate)/(2*len(train))
		weights[i] = weight



# Performs gradient descent on 4th Power Error Cost function
def fitCurve4thPower(train, weights, learning_rate):
	for i in range(len(deg)):
		# Generate the feature vector and get the weight vector in proper format
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(train[k][0],j) for j in range(deg[i]+1)] for k in range(len(train))]

		# dtype=float to prevent the overshooting of the weight parameters
		x = np.array(x,dtype=np.float128)
		x = np.transpose(x)
		weight = np.array(weight, dtype=np.float128)

		# Update the weight for n_iter iterations
		for n in range(n_iter):
			h = np.transpose(weight).dot(x)
			h = np.array(h,dtype=np.float128)
			y = np.array(zip(*train)[1],dtype=np.float128)
			y = y.reshape(h.shape)
			err = np.zeros(weight.shape,dtype=np.float128)
			err = np.transpose(np.power((h-y),3,dtype=np.float128).dot(np.transpose(x)))
			weight -= (2 * err * learning_rate)/len(train)
		weights[i] = weight


# Function that estimates the root mean squared error given the weights and the data
def estimateError(data, weights, err):
	for i in range(len(deg)):
		# Initialize the weights and the feature vector in proper order
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(data[k][0],j) for j in range(deg[i]+1)] for k in range(len(data))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)

		# Calculate the root mean squared error
		h = np.matmul(np.transpose(weight), x)
		y = np.array(zip(*data)[1])
		y = y.reshape(h.shape)
		a = (h - y).reshape(-1)*(h - y).reshape(-1)
		err[i] = np.sum(a)
		err[i] /= len(x)
		err[i] = math.sqrt(err[i])




# Initialization of different values
n_sample = 100
learning_rate = [0.025, 0.05, 0.1, 0.2, 0.5]
deg = [i for i in range(1,10)]
sample = []
train = []
test = []

weights_square = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_square = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_square = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]

weights_abs = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_abs = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_abs = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]

weights_power4 = [[[0 for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_power4 = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_power4 = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]


# Generate and split the sample intrain and test set. The samples are stored in a 2D matrix of shape (len(n_samples),n_samples)
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

x = [sample[i][0] for i in range(len(sample))]
y = [sample[i][1] for i in range(len(sample))]

splitSample(sample, train, test)
train, test = train[0],test[0]
print("\nDataset split into 80% training data and 20% test data")


# For each learning rate
for i in range(len(learning_rate)):

	# Fit the regression curve with various Cost function and estimate the squared error on test set
	print("\nFitting a curve of " + str(n_sample) + " samples (learning rate = " + str(learning_rate[i]) + ") using linear regression...")
	fitCurveSquared(train, weights_square[i], learning_rate[i])
	estimateError(test, weights_square[i], err_test_square[i])

	fitCurveAbs(train, weights_abs[i], learning_rate[i])
	estimateError(test, weights_abs[i], err_test_abs[i])

	fitCurve4thPower(train, weights_power4[i], learning_rate[i])
	estimateError(test, weights_power4[i], err_test_power4[i])

	print("")
	for m in range(len(deg)):
		print("Degree of Polynomial n: " + str(deg[m]))
		print("Squared Cost Weights Learned: ")
		print(np.transpose(weights_square[i][m]))
		print("Squared Cost Test Error: " + str(err_test_square[i][m]))
		print("Absolute Cost Weights Learned: ")
		print(np.transpose(weights_abs[i][m]))
		print("Absolute Cost Test Error: " + str(err_test_abs[i][m]))
		print("4th Power Cost Weights Learned: ")
		print(np.transpose(weights_power4[i][m]))
		print("4th Power Cost Test Error: " + str(err_test_power4[i][m]))
		print("")
	print("")

		# Generates the regression curve and plots it
		x1 = np.linspace(0, 1, 100)
		y1,y2,y3 = [],[],[]
		for k in x1:			# finds the value of the function using x values and the parameters of the function
			a1,a2,a3 = 0,0,0
			for j in range(deg[m]+1):
				a1 += weights_abs[i][m][j]*math.pow(k,j)
				a2 += weights_abs[i][m][j]*math.pow(k,j)
				a3 += weights_abs[i][m][j]*math.pow(k,j)
			y1.append(a1)
			y2.append(a2)
			y3.append(a3)

		# Plots the dataset along with the curve fir for each cost function
		plt.figure()
		plt.scatter(x,y)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("Squared Error Cost Function, alpha = "+str(learning_rate[i])+", DEGREE = "+str(deg[m]))
		plt.plot(x1, y1, "k")
		plt.savefig("Squared Error Cost Function alpha "+str(learning_rate[i])+" Degree "+str(deg[m])+".png")

		plt.figure()
		plt.scatter(x,y)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("Absolute Error Cost Function, alpha = "+str(learning_rate[i])+", DEGREE = "+str(deg[m]))
		plt.plot(x1, y1, "k")
		plt.savefig("Absolute Error Cost Function alpha "+str(learning_rate[i])+" Degree "+str(deg[m])+".png")

		plt.figure()
		plt.scatter(x,y)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("4th Power Error Cost Function, alpha = "+str(learning_rate[i])+", DEGREE = "+str(deg[m]))
		plt.plot(x1, y1, "k")
		plt.savefig("4thPower Error Cost Function alpha "+str(learning_rate[i])+" Degree "+str(deg[m])+".png")


# Plots the RMSE vs Learning Rate curve
for i in range(len(deg)):
	plt.figure()
	plt.plot(learning_rate, zip(*err_test_square)[i], "b")
	plt.plot(learning_rate, zip(*err_test_abs)[i], "r")
	plt.plot(learning_rate, zip(*err_test_power4)[i], "g")
	plt.xlabel("Learning Rate (alpha)")
	plt.ylabel("Error")
	plt.title("RMSE vs LEARNING RATE, DEGREE = "+str(deg[i]))
	plt.legend(["Squared Error Cost Funtion","Absolute Error Cost Function","4th Power Error Cost Function"])
	plt.savefig("Variation with Learning Rate, Degree "+str(deg[i]))
# plt.show()

print("\n\nOver..")