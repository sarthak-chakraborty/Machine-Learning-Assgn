import numpy as np 
import pandas as pd 
import nltk
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import random
from math import e
import matplotlib.pyplot as plt 


batch_size = 32
n_hidden_layer = 1
n_nodes = [100]
n_nodes_total = [0] + n_nodes
learning_rate = 0.1
n_epochs = 30
n_nodes_out = 1
thresh = 0.5


def preprocess(x_train, y_train, x_test, y_test):
	data = []
	d = []
	stopwords = []
	label = []
	with open("../Assignment_4_data.txt") as f:
		for line in f:
			a = line.split('\t')
			label.append(str(a[0]))
			d.append(str(a[1]))

	for i in range(len(label)):
		label[i] = 0 if(label[i]=='ham') else 1

	with open("../NLTK's list of english stopwords") as f:
		stopwords = [line.strip().split('\n')[0] for line in f]

	tokenized_data = []
	delim = " ", "\t", "\n", "\.", ",", ":", "-", "\?", "\'", "\"", "(", ")", "[", "]", "{", "}", "!", "\*", "\\", "/"
	regexPattern = '|'.join(map(re.escape, delim))
	for x in d:
		a = re.split(regexPattern, x)
		del a[-1]
		a = [y.lower() for y in a]
		tokenized_data.append(a)

	data_stopwords = []
	for i in range(len(tokenized_data)):
		a = [word for word in tokenized_data[i] if not word in stopwords]
		data_stopwords.append(a)

	ps = PorterStemmer()
	for sent in data_stopwords:
		a = list({str(ps.stem(word)) for word in sent})
		a.sort()
		data.append(a)
		
	tokens = set()
	for sample in data:
		tokens = tokens.union(set(sample))
	tokens = list(tokens)
	tokens.sort()
	n_nodes_total[0] = len(tokens)

	train = []
	for i in range(len(data)):
		bog = np.array([0 for k in range(len(tokens))])
		for word in data[i]:
			index = tokens.index(word) if word in tokens else -1
			if index is not -1:
				bog[index] = 1
		train.append(bog)


	for i in range(len(train)):
		x = np.random.uniform(0,1)
		if(x<0.8):
			x_train.append(train[i])
			y_train.append(label[i])
		else:
			x_test.append(train[i])
			y_test.append(label[i])



def data_loader(x_train, y_train):
	X = [(x_train[i], y_train[i]) for i in range(len(x_train))]
	random.shuffle(X)
	x_load, y_load = [], []
	for i in range(int(len(X)/batch_size)):
		x_load.append([b[0] for b in X[0:batch_size]])
		y_load.append([b[1] for b in X[0:batch_size]])
		del X[:batch_size]
	x_load.append([b[0] for b in X])
	y_load.append([b[1] for b in X])

	return np.array(x_load), np.array(y_load)
			


def relu(x):
	return np.array([max(0,i) for i in x])


def relu_derivative(x):
	a = 0 if x<=0 else 1
	return a


def sigmoid(x):
	f = []
	for i in x:
		a = 1.0/(1 + e**(-i))
		f.append(a)

	return np.array(f)


def sigmoid_derivative(x):
	s = 1.0/(1 + e**(-x))
	return s*(1-s)


def weight_initializer(W, in_dim, out_dim):
	W[0] = np.array([[np.random.uniform(-0.01,0.01) for i in range(n_nodes[0])] for j in range(in_dim)])
	W[-1] = np.array([[np.random.uniform(-0.01,0.01) for i in range(out_dim)] for j in range(n_nodes[n_hidden_layer-1])])
	for i in range(len(W)-2):
		W[i+1] = [[np.random.uniform(-0.01,0.01) for j in range(n_nodes[i+1])] for k in range(n_nodes[i])]
	return W


def bias_initializer(B, out_dim):
	B[-1] = [np.random.uniform(0,0.0001) for i in range(out_dim)]
	for i in range(len(B)-1):
		B[i] = [np.random.uniform(0,0.0001) for j in range(n_nodes[i])]
	return B




def forward(weights, bias, x_train):

	inp = []
	out = []
	
	pred = []
	for i in range(len(x_train)):
		hidden_state = x_train[i]
		a, b = [], []

		for j in range(n_hidden_layer+1):
			b.append(hidden_state)
			a.append(np.dot(hidden_state, weights[j]) + bias[j])
			if(j == n_hidden_layer):
				hidden_state = sigmoid(np.dot(hidden_state, weights[j]) + bias[j])
			else:
				hidden_state = relu(np.dot(hidden_state, weights[j]) + bias[j])

		inp.append(a)
		out.append(b)
		pred.append(hidden_state)
	return np.array(pred), np.array(inp), np.array(out)





def backward(weights, bias, inputs, outputs, pred, actual):
	n_samples = len(inputs)
	

	delta_error = [[-1.0/pred[i][j] if(actual[i]==1) else 1.0/(1 - pred[i][j]) for j in range(n_nodes_out)] for i in range(n_samples)]
	delta_error = np.array(delta_error)

	delta = []
	for i in range(n_samples):
		a = []
		for j in range(n_nodes_out):
			inp = sigmoid_derivative(inputs[i][n_hidden_layer][j])
			a.append(delta_error[i][j] * inp)
		delta.append(a)
	delta = np.array(delta)


	for i in range(n_hidden_layer+1):
		nebla_weights = np.dot(delta.T, np.array(list(zip(*outputs))[-1-i]))/n_samples
		nebla_bias = np.array([np.sum(delta, axis=0)]).reshape(-1)/n_samples

		if(i != n_hidden_layer):
			b = np.dot(delta, weights[-1-i].T)
			delta = []
			for j in range(n_samples):
				a = []
				for k in range(n_nodes_total[-1-i]):
					inp = relu_derivative(inputs[j][n_hidden_layer-1-i][k])
					a.append(b[j][k] * inp)
				delta.append(a)
			delta = np.array(delta)


		weights[-1-i] -= learning_rate*nebla_weights.T
		bias[-1-i] -= learning_rate*nebla_bias.T	
		
	return weights, bias



def get_train_accuracy(weights, bias, X, Y):
	count = 0
	N = 0
	predicted = []
	actual = []
	for i in range(len(X)):
		pred, inp, out = forward(weights, bias, X[i])
		predicted = predicted + list(list(zip(*pred))[0])
		actual = actual + Y[i]
		p = [1 if pred[i][0] > thresh else 0 for i in range(len(pred))]
		for j in range(len(p)):
			N += 1
			if(p[j] == Y[i][j]):
				count += 1

	error = [-1.0*np.log(predicted[i]) if actual[i]==1 else -1.0*np.log(1 - predicted[i]) for i in range(len(actual))]
	error = np.sum(error)/len(actual)

	
	return count/N, error



def training(weights, bias, X_train, Y_train, X_test, Y_test):
	accuracy_train, error_train, accuracy_test = [], [], []
	for i in range(n_epochs):
		for j in range(len(X_train)):
			pred, inp, out = forward(weights, bias, X_train[j])
			weights, bias = backward(weights, bias, inp, out, pred, Y_train[j])

		acc_train, err = get_train_accuracy(weights, bias, X_train, Y_train)
		accuracy_train.append(1-acc_train)
		error_train.append(err)
		acc_test = get_test_accuracy(weights, bias, X_test, Y_test)
		accuracy_test.append(1-acc_test)
		print("EPOCH ",i+1,"\tTrain Error: ",(1-acc_train),"\tLoss: ",err,"\tTest Error: ",(1-acc_test))


	print("\nTrain Set accuracy: ",1-accuracy_train[-1])
	return weights, bias, accuracy_train, error_train, accuracy_test



def get_test_accuracy(weights, bias, X, Y):
	count = 0
	N = 0
	pred, inp, out = forward(weights, bias, X)
	p = [1 if pred[i][0] > thresh else 0 for i in range(len(pred))]
	for j in range(len(p)):
		N += 1
		if(p[j] == Y[j]):
			count += 1

	return count/N





x_train = []
y_train = []
x_test = []
y_test = []
preprocess(x_train, y_train, x_test, y_test)

np.save("x_train", x_train)
np.save("y_train", y_train)
np.save("x_test", x_test)
np.save("y_test", y_test)


# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")
# x_test = np.load("x_test.npy")
# y_test = np.load("y_test.npy")


X_train, Y_train = data_loader(x_train, y_train)


weights = [[] for i in range(n_hidden_layer+1)]
bias = [[] for i in range(n_hidden_layer+1)]
weights = np.array(weight_initializer(weights, len(x_train[0]), n_nodes_out)) 
bias = np.array(bias_initializer(bias, n_nodes_out))


np.seterr(all='ignore')
weights, bias, accuracy_train, error_train, accuracy_test = training(weights, bias, X_train, Y_train, x_test, y_test)


epoch = [i for i in range(n_epochs)]

plt.figure()
plt.plot(epoch, accuracy_train)
plt.plot(epoch, accuracy_test, color='r')
plt.title("Error vs Number of Epochs")
plt.ylabel("Error")
plt.xlabel("Number of Epochs")
plt.legend(["Train", "Test"])
# plt.savefig("Error vs Epoch Plot.png")

plt.figure()
plt.plot(epoch, error_train)
plt.title("Loss Function vs Number of Epochs")
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
# plt.savefig("Loss Function vs Epoch(training) Plot.png")

plt.show()


acc = get_test_accuracy(weights, bias, x_test, y_test)
print("Test Set Accuracy: ", acc)
