import numpy as np 
import matplotlib.pyplot as plt
import math

n_sample = 100
learning_rate = 0.05
n_iter = 100
deg = [i for i in range(1,10)]


def genSample(sample):
	for i in range(n_sample):
		x = np.random.uniform(0,1)
		y = np.sin(2*np.pi*x)
		y += np.random.normal(loc=0,scale=0.3)
		sample.append((x,y))
	sample = np.array(sample)


def splitSample(sample, train, test):
	for i in sample:
		n = np.random.uniform(0,1)
		if(n>0.8):
			test.append(i)
		else:
			train.append(i)


def fitCurve(train, weights):
	for i in deg:
		weight = np.reshape(np.array(weights[i-1]),(i+1,1))
		x = [[math.pow(train[k][0],j) for j in range(i+1)] for k in range(len(train))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(i+1,1))

		for n in range(n_iter):
			for k in range(len(weight)):
				err = 0.0
				for j in range(len(x)):
					h = np.matmul(np.transpose(weight), x[j])
					err += (h - train[j][1]) * x[j][k]
				weight[k] -= (err[0][0] * learning_rate)/len(x);
		weights[i-1] = weight



def estimateError(data, weights, err):
	for i in deg:
		weight = np.reshape(np.array(weights[i-1]),(i+1,1))
		x = [[math.pow(data[k][0],j) for j in range(i+1)] for k in range(len(data))]
		for j in range(len(x)):
			x[j] = np.reshape(np.array(x[j]),(i+1,1))

		for j in range(len(x)):
			h = np.matmul(np.transpose(weight), x[j])
			err[i-1] += math.pow((h - data[j][1]), 2)

		err[i-1] /= 2*len(x)




sample = []
train = []
test = []
weights = [[np.random.uniform(0,1) for i in range(1+j)] for j in deg]
err_train = [0 for i in range(len(deg))]
err_test = [0 for i in range(len(deg))]

genSample(sample)
splitSample(sample, train, test)
fitCurve(train, weights)
estimateError(train, weights, err_train)
estimateError(test, weights, err_test)

plt.figure()
plt.plot(deg, err_train, "r")
plt.plot(deg, err_test, "k")
plt.show()

x = [sample[i][0] for i in range(len(sample))]
y = [sample[i][1] for i in range(len(sample))]

# for i in deg:
# 	plt.figure()
# 	plt.scatter(x,y)
# 	x1 = np.linspace(0, 1, 100)
# 	y1 = []
# 	for k in x1:
# 		a = 0
# 		for j in range(i+1):
# 			a += weights[i-1][j]*math.pow(k,j)
# 		y1.append(a)
# 	plt.plot(x1, y1, "k")

# plt.show()
