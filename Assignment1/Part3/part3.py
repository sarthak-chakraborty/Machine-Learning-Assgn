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

		x = np.array(x)
		x = np.transpose(x)
		weight = np.array(weight)
		for n in range(n_iter):
			h = np.transpose(weight).dot(x)
			y = np.array(zip(*train)[1])
			y = y.reshape(h.shape)
			err = np.transpose((h-y).dot(np.transpose(x)))
			weight -= (err * learning_rate)/len(train)
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



n_sample = [10, 100, 1000, 4000, 7000, 10000]
learning_rate = 0.05
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
	fitCurveSquared(train[i], weights[i])
	print("Curve fitting over")
	estimateError(train[i], weights[i], err_train[i])
	estimateError(test[i], weights[i], err_test[i])

mindeg = err_test[len(n_sample)/2].index(min(err_test[len(n_sample)/2]))
tr_err = zip(*err_train)[mindeg]
ts_err = zip(*err_test)[mindeg]

plt.figure()
plt.plot(n_sample, tr_err, color="r")
plt.plot(n_sample, ts_err, color="k")
plt.xlabel("Number of Samples")
plt.ylabel("Error Value")
plt.title("LEARNING CURVE")
plt.legend(["Train Error","Test Error"])
plt.savefig("Learning Curve")
# plt.show()
print("Over..")