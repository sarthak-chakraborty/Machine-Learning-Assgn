import numpy as np 
import matplotlib.pyplot as plt
import math


n_iter = 5000


def genSample(sample):
	a = np.random.uniform(0,1,n_sample)
	b = np.sin(2*np.pi*np.array(a)).reshape(-1)
	b += np.random.normal(loc=0,scale=0.3,size=n_sample)
	sample.append([(a[i], b[i]) for i in range(len(a))])



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



def fitCurveAbs(train, weights, learning_rate):
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
		err[i] /= len(x)
		err[i] = math.sqrt(err[i])



n_sample = 100
learning_rate = [0.025, 0.05, 0.1, 0.2, 0.5]
deg = [8]
sample = []
train = []
test = []
weights = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]

print("Generating " + str(n_sample) + " samples...")
genSample(sample)
print("Dataset Generated")
sample = sample[0]

x = [sample[i][0] for i in range(len(sample))]
y = [sample[i][1] for i in range(len(sample))]

splitSample(sample, train, test)
train, test = train[0],test[0]
print("\nDataset split into 80% training data and 20% test data")

for i in range(len(learning_rate)):
	print("\nFitting a curve of " + str(n_sample) + " samples (learning rate = " + str(learning_rate[i]) + ") using linear regression...")
	fitCurveAbs(train, weights[i], learning_rate[i])
	estimateError(test, weights[i], err_test[i])

	print("\nPlots the graph of the dataset along with the curve fit")

	for m in range(len(deg)):
		plt.figure()
		plt.scatter(x,y)
		x1 = np.linspace(0, 1, 100)
		y1 = []
		for k in x1:
			a = 0
			for j in range(deg[m]+1):
				a += weights[i][m][j]*math.pow(k,j)
			y1.append(a)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("REGRESSION LINE (DEGREE = "+str(deg[m])+")")
		plt.plot(x1, y1, "k")
plt.show()