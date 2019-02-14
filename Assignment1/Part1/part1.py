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



n_sample = 10
learning_rate = 0.05
deg = [i for i in range(1,10)]
sample= []
train = []
test = []
weights = [[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))]
err_train = [0 for i in range(len(deg))]
err_test = [0 for i in range(len(deg))]

print("Generating " + str(n_sample) + " samples...")
genSample(sample)
print("Dataset Generated")
sample = sample[0]

splitSample(sample, train, test)
train = train[0]
test = test[0]
print("\nDataset split into 80% training data and 20% test data")

print("\nFitting a curve of " + str(n_sample) + " samples using linear regression...")
fitCurveSquared(train, weights)
print("Curve fitting over")
estimateError(train, weights, err_train)
estimateError(test, weights, err_test)

np.save("dataset",sample)
np.save("weights", weights)
np.save("train error",err_train)
np.save("test error",err_test)

print("Over..")