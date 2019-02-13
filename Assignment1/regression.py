import numpy as np 
import matplotlib.pyplot as plt
import math


n_iter = 5000


def genSample(sample):
	for j in range(len(n_sample)):
		a = np.random.uniform(0,1,n_sample[j])
		b = np.sin(2*np.pi*np.array(a)).reshape(-1)
		b += np.random.normal(loc=0,scale=0.3,size=n_sample[j])
		sample.append([(a[i], b[i]) for i in range(len(a))])



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



def fitCurveSquared(train, weights):
	for i in range(len(deg)):
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(train[k][0],j) for j in range(deg[i]+1)] for k in range(len(train))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)
		for n in range(n_iter):
			h = np.matmul(np.transpose(weight),x)
			y = np.array(zip(*train)[1])
			y = y.reshape(h.shape)
			for k in range(len(weight)):
				err = (h - y).reshape((1,-1)) * np.array(zip(*np.transpose(x))[k])
				err = np.sum(err)
				weight[k] -= (err * learning_rate)/len(x);
		weights[i] = weight



def fitCurveAbs(train, weights):
	for i in range(len(deg)):
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(train[k][0],j) for j in range(deg[i]+1)] for k in range(len(train))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)
		for n in range(n_iter):
			h = np.matmul(np.transpose(weight),x)
			y = np.array(zip(*train)[1])
			y = y.reshape(h.shape)
			a = (h - y).reshape(-1)
			a = a/abs(a)
			for k in range(len(weight)):
				err = a * np.array(zip(*np.transpose(x))[k]).reshape(-1)
				err = np.sum(err)
				weight[k] -= (err * learning_rate)/(2*len(x));
		weights[i] = weight



def fitCurve4thPower(train, weights):
	for i in range(len(deg)):
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(train[k][0],j) for j in range(deg[i]+1)] for k in range(len(train))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)
		for n in range(n_iter):
			h = np.matmul(np.transpose(weight),x)
			y = np.array(zip(*train)[1])
			y = y.reshape(h.shape)
			for k in range(len(weight)):
				err = (h-y).reshape((1,-1))*(h-y).reshape((1,-1))*(h-y).reshape((1,-1))*np.array(zip(*np.transpose(x))[k])
				err = np.sum(err)
				weight[k] -= 2*(err * learning_rate)/len(x);
		weights[i] = weight



def estimateError(data, weights, err):
	for i in range(len(deg)):
		weight = np.reshape(np.array(weights[i]),(deg[i]+1,1))
		x = [[math.pow(data[k][0],j) for j in range(deg[i]+1)] for k in range(len(data))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(deg[i]+1,1))

		x = np.array(x)
		weight = np.array(weight)
		h = np.matmul(np.transpose(weight), x)
		y = np.array(zip(*data)[1])
		y = y.reshape(h.shape)
		a = (h - y).reshape(-1)*(h - y).reshape(-1)
		err[i] = np.sum(a)
		err[i] /= 2*len(x)




print("\tWELCOME to the ML Assignment 1")
while(1):
	mindeg = 1
	print("\nEnter the part that you choose to perform...")
	print(" 1. Synthetic Data Generation and Simple Curve Fitting (PRESS 1)")
	print(" 2. Visualization of the Dataset and the fitted curves (PRESS 2)")
	print(" 3. Experimenting with larger training set (PRESS 3)")
	print(" 4. Experimenting with cost functions (PRESS 4)")
	print(" 5. Exit (PRESS 5)")
	c = input("Enter Choice: ")

	if(c==5):
		print("Thank you.. Exiting..")
		break

	if(c==1):
		print("\na) Synthetically generates 10 samples of data")
		print("b) Split the data into training and test set")
		print("c) Fits a curve that minimizes squared error cost function using linear regression. Degree of polynomial = 1 to 9. Learning rate = 0.05")
		
		n_sample = [100]
		learning_rate = 0.05
		deg = [i for i in range(1,10)]
		sample= []
		train = []
		test = []
		weights = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(n_sample))]
		err_train = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		err_test = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		
		print("Generating " + str(n_sample) + " samples...")
		genSample(sample)
		print("Dataset Generated")
		
		splitSample(sample, train, test)
		print("\nDataset split into 80% training data and 20% test data")
		
		for i in range(len(n_sample)):
			print("\nFitting a curve of " + str(n_sample[i]) + " samples using linear regression...")
			fitCurveSquared(train[i], weights[i])
			print("Curve fitting over")
			estimateError(train[i], weights[i], err_train[i])
			estimateError(test[i], weights[i], err_test[i])
		print("Over..")

	if(c==2):
		print("\nPlots the graph of the dataset along with the curve fit")
		x = [sample[0][i][0] for i in range(len(sample[0]))]
		y = [sample[0][i][1] for i in range(len(sample[0]))]

		for i in range(len(deg)):
			plt.figure()
			plt.scatter(x,y)
			x1 = np.linspace(0, 1, 100)
			y1 = []
			for k in x1:
				a = 0
				for j in range(deg[i]+1):
					a += weights[0][i][j]*math.pow(k,j)
				y1.append(a)
			plt.xlabel("X")
			plt.ylabel("Y")
			plt.title("REGRESSION LINE (DEGREE = "+str(deg[i])+")")
			plt.plot(x1, y1, "k")

		plt.figure()
		plt.plot(deg, err_train[0], "r")
		plt.plot(deg, err_test[0], "k")
		plt.xlabel("Degree of Regression Line")
		plt.ylabel("Error Value")
		plt.title("ERROR vs DEGREE of Regression Line")
		plt.legend(["Train","Test"])
		plt.xticks(deg)
		plt.show()
		print("Over..")
		mindeg = min(err_test[0])

	if(c==3):
		n_sample = [10, 100, 1000, 10000]
		learning_rate = 0.05
		deg = [mindeg]
		sample = []
		train = []
		test = []
		weights = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(n_sample))]
		err_train = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		err_test = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		
		print("Generating " + str(n_sample) + " samples...")
		genSample(sample)
		print("Dataset Generated")
		splitSample(sample, train, test)
		print("\nDataset split into 80% training data and 20% test data")
		
		for i in range(len(n_sample)):
			print("\nFitting a curve of " + str(n_sample[i]) + " samples using linear regression...")
			fitCurveSquared(train[i], weights[i])
			print("Curve fitting over")
			estimateError(train[i], weights[i], err_train[i])
			estimateError(test[i], weights[i], err_test[i])

		plt.figure()
		plt.plot(n_sample, err_train, color="r")
		plt.plot(n_sample, err_test, color="k")
		plt.xlabel("Number of Samples")
		plt.ylabel("Error Value")
		plt.title("LEARNING CURVE")
		plt.legend(["Train Error","Test Error"])
		plt.show()
		print("Over..")

	if(c==4):
		n_sample = [100]
		learning_rate = 0.05#[0.025, 0.05, 0.1, 0.2, 0.5]
		deg = [i for i in range(1,10)]
		sample = []
		train = []
		test = []
		weights = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(n_sample))]
		err_train = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		err_test = [[0 for i in range(len(deg))] for k in range(len(n_sample))]
		
		print("Generating " + str(n_sample) + " samples...")
		genSample(sample)
		print("Dataset Generated")
		splitSample(sample, train, test)
		print("\nDataset split into 80% training data and 20% test data")
		for i in range(len(n_sample)):
			print("\nFitting a curve of " + str(n_sample[i]) + " samples using linear regression...")
			fitCurveAbs(train[i], weights[i])

		print("\nPlots the graph of the dataset along with the curve fit")
		x = [sample[0][i][0] for i in range(len(sample[0]))]
		y = [sample[0][i][1] for i in range(len(sample[0]))]

		for i in range(len(deg)):
			plt.figure()
			plt.scatter(x,y)
			x1 = np.linspace(0, 1, 100)
			y1 = []
			for k in x1:
				a = 0
				for j in range(deg[i]+1):
					a += weights[0][i][j]*math.pow(k,j)
				y1.append(a)
			plt.xlabel("X")
			plt.ylabel("Y")
			plt.title("REGRESSION LINE (DEGREE = "+str(deg[i])+")")
			plt.plot(x1, y1, "k")
		plt.show()