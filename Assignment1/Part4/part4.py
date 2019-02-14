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



def fitCurveSquared(train, weights, learning_rate):
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



def fitCurveAbs(train, weights, learning_rate):
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
			a = (h - y)/abs(h-y)
			err = np.transpose(a.dot(np.transpose(x)))
			weight -= (err * learning_rate)/(2*len(train))
		weights[i] = weight



def fitCurve4thPower(train, weights, learning_rate):
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
			a = h-y
			err = np.transpose(np.power(a,3).dot(np.transpose(x)))
			weight -= (2 * err * learning_rate)/len(train)
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

weights_square = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_square = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_square = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]

weights_abs = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_abs = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_abs = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]

weights_power4 = [[[np.random.uniform(0,1) for i in range(1+deg[j])] for j in range(len(deg))] for k in range(len(learning_rate))]
err_train_power4 = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]
err_test_power4 = [[0 for i in range(len(deg))] for k in range(len(learning_rate))]


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
	fitCurveSquared(train, weights_square[i], learning_rate[i])
	estimateError(test, weights_square[i], err_test_square[i])

	fitCurveAbs(train, weights_abs[i], learning_rate[i])
	estimateError(test, weights_abs[i], err_test_abs[i])

	fitCurve4thPower(train, weights_power4[i], learning_rate[i])
	estimateError(test, weights_power4[i], err_test_power4[i])

	# print("\nPlots the graph of the dataset along with the curve fit")

	# for m in range(len(deg)):
	# 	plt.figure()
	# 	plt.scatter(x,y)
	# 	x1 = np.linspace(0, 1, 100)
	# 	y1 = []
	# 	for k in x1:
	# 		a = 0
	# 		for j in range(deg[m]+1):
	# 			a += weights_abs[i][m][j]*math.pow(k,j)
	# 		y1.append(a)
	# 	plt.xlabel("X")
	# 	plt.ylabel("Y")
	# 	plt.title("REGRESSION LINE (DEGREE = "+str(deg[m])+")")
	# 	plt.plot(x1, y1, "k")

plt.figure()
plt.plot(learning_rate, err_test_square, "b")
plt.plot(learning_rate, err_test_abs, "r")
plt.plot(learning_rate, err_test_power4, "g")
plt.xlabel("Learning Rate (alpha)")
plt.ylabel("Error")
plt.title("RMSE vs LEARNING RATE")
plt.legend(["Squared Cost Funtion","Absolute Cost Function","Power 4 Cost Function"])
plt.savefig("Variation with Learning Rate")
# plt.show()