import numpy as np 
import pandas as pd 
import nltk
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import random


batch_size = 512
n_hidden_layer = 1
n_nodes = [100]
learning_rate = 0.1
n_epochs = 1
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
	delim = [' ', '\t' , '\n' , '.' , ',' , ':' , '-']
	for x in d:
		a = re.split('\W+', x)
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
	f = 0 if x<=0 else 1
	return f



def weight_initializer(W, in_dim, out_dim):
	W[0] = np.array([[np.random.uniform(0,1) for i in range(n_nodes[0])] for j in range(in_dim)])
	W[-1] = np.array([[np.random.uniform(0,1) for i in range(out_dim)] for j in range(n_nodes[n_hidden_layer-1])])
	for i in range(len(W)-2):
		w[i+1] = [[np.random.uniform(0,1) for j in range(n_nodes[i+1])] for k in range(n_nodes[i])]
	return W


def bias_initializer(B, out_dim):
	B[-1] = [np.random.uniform(0,1) for i in range(out_dim)]
	for i in range(len(B)-1):
		B[i] = [np.random.uniform(0,1) for j in range(n_nodes[i])]
	return B




def forward(weights, bias, x_train):
	pred = []
	for sample in x_train:
		hidden_state = sample
		for i in range(n_hidden_layer+1):
			hidden_state = relu(np.dot(hidden_state, weights[i]) + bias[i])
		pred.append(0 if hidden_state[0]>thresh else 1)

	return pred





def backward(weights, bias, pred, label):
	# return
	error = np.sum([-np.log2(pred[i]+np.finfo(float).eps) if label[i] == 1 else -np.log2(1-pred[i]++np.finfo(float).eps) for i in range(len(label))])

	# for i in range(len(weights))






# x_train = []
# y_train = []
# x_test = []
# y_test = []
# preprocess(x_train, y_train, x_test, y_test)

# np.save("x_train", x_train)
# np.save("y_train", y_train)
# np.save("x_test", x_test)
# np.save("y_test", y_test)


x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


X_train, Y_train = data_loader(x_train, y_train)


weights = [[] for i in range(n_hidden_layer+1)]
bias = [[] for i in range(n_hidden_layer+1)]
weights = np.array(weight_initializer(weights, len(x_train[0]), 1)) # 1 here says number of output layer
bias = np.array(bias_initializer(bias, 1))



for i in range(n_epochs):
	for j in range(len(X_train)):
		pred = np.array(forward(weights, bias, X_train[j]))
		# weights, bias = backward(weights, bias, pred, Y_train[j])
		