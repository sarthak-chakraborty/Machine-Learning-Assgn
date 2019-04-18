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
n_nodes_total = [0] + n_nodes
learning_rate = 0.1
n_epochs = 1
n_nodes_out = 1
thresh = 500


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
	delim = " ", "\t", "\n", ".", ",", ":", "-", "?"
	regexPattern = '|'.join(map(re.escape, delim))
	for x in d:
		a = re.split(regexPattern, x)
		# a = re.split('\W+', x)
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
	print(len(tokens))
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



def weight_initializer(W, in_dim, out_dim):
	W[0] = np.array([[np.random.uniform(0,1) for i in range(n_nodes[0])] for j in range(in_dim)])
	W[-1] = np.array([[np.random.uniform(0,1) for i in range(out_dim)] for j in range(n_nodes[n_hidden_layer-1])])
	for i in range(len(W)-2):
		W[i+1] = [[np.random.uniform(0,1) for j in range(n_nodes[i+1])] for k in range(n_nodes[i])]
	return W


def bias_initializer(B, out_dim):
	B[-1] = [np.random.uniform(0,1) for i in range(out_dim)]
	for i in range(len(B)-1):
		B[i] = [np.random.uniform(0,1) for j in range(n_nodes[i])]
	return B




def forward(weights, bias, x_train):	# x_train has 512 numbers of 7306 length vector 
	# inp is a 3D array of shape (n_samples, n_layers+1, n_nodes)
	inp = [[[] for i in range(n_hidden_layer+1)] for j in range(len(x_train))]	# len(x_train) is batch_size(512)
	out = [[[] for i in range(n_hidden_layer+1)] for j in range(len(x_train))]
	# print(inp)
	pred = []
	for i in range(len(x_train)):
		hidden_state = x_train[i]

		for j in range(n_hidden_layer+1):
			out[i][j] = hidden_state
			inp[i][j] = np.dot(hidden_state, weights[j]) + bias[j]
			# print(hidden_state.shape)
			# print(weights[j].shape)
			# print("HELLO: ",np.dot(hidden_state, weights[j]).shape)
			# print("")
			z = np.dot(hidden_state, weights[j]) + bias[j]
			hidden_state = relu(np.dot(hidden_state, weights[j]) + bias[j])
			# print(hidden_state)
		# print(hidden_state)
		pred.append([0 if hidden_state[i]>thresh else 1 for i in range(len(hidden_state))]) #pred is a 2D array (n_sample, n_out_nodes)
	print(pred)
	return np.array(pred), np.array(inp), np.array(out)





def backward(weights, bias, inputs, outputs, pred, actual):
	n_samples = len(inputs)
	
	delta_error = [[-1.0/(pred[i][j] + np.finfo(float).eps) if(actual[i]==1) else 1.0/(1 - pred[i][j] + np.finfo(float).eps) for j in range(n_nodes_out)] for i in range(n_samples)]
	delta_error = np.array(delta_error)
	# print(delta_error.shape)
	

	delta = [[] for i in range(n_samples)]
	for i in range(n_samples):
		a = []
		for j in range(n_nodes_out):
			inp = relu_derivative(inputs[i][n_hidden_layer][j])
			a.append(delta_error[i][j] * inp)
		delta[i] = a
	delta = np.array(delta)

	# print(delta.shape)

	for i in range(n_hidden_layer+1):
		nebla_weights = np.matmul(delta.T, np.array(list(zip(*outputs))[-1-i]))
		nebla_bias = np.array([np.sum(delta, axis=0)]).reshape(-1)

		# print("Nebla Weight : ",nebla_weights.shape)
		# print("Nebla Bias: ", nebla_bias.T.shape)
		# print(np.array(weights[-1-i]).shape)
		# print(np.array(bias[-1-i]).shape)

		if(i != n_hidden_layer):
			b = np.dot(delta, weights[-1-i].T)
			# print("b: ",b.shape)
			# print(np.array(list(zip(*inputs))[-1-i]).shape)
			delta = [[] for i in range(n_samples)]
			for j in range(n_samples):
				a = []
				for k in range(n_nodes_total[-1-i]):
					inp = relu_derivative(inputs[j][n_hidden_layer-1-i][k])
					a.append(b[j][k] * inp)
				delta[j] = a
			delta = np.array(delta)
			# print("delta : ",delta.shape)

		# delta = np.matmul(np.array(list(zip(*inputs))[-1-i]), b)

		weights[-1-i] -= learning_rate*nebla_weights.T
		bias[-1-i] -= learning_rate*nebla_bias.T	
		
	return weights, bias



def get_accuracy(weights, bias, X_train, Y_train):
	count = 0
	for i in range(len(X_train)):
		pred, inp, out = forward(weights, bias, X_train[i])
		# print(pred)
		# print("")
		for j in pred:
			if(j[0] == Y_train[j]):
				count += 1

	print(count)
	return count/len(Y_train)



def training(weights, bias, X_train, Y_train):
	for i in range(n_epochs):
		print("EPOCH ",i+1)
		print("=========")
		updated_weights, updated_bias = weights, bias
		for j in range(len(X_train)):
			print("Batch ",j+1)
			pred, inp, out = forward(updated_weights, updated_bias, X_train[j])
			w, b = backward(updated_weights, updated_bias, inp, out, pred, Y_train[j])
			updated_weights = w
			updated_bias = b
		print("")

		accuracy = get_accuracy(weights, bias, X_train, Y_train)
		print("Accuracy: ",accuracy)

	return weights, bias






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
weights = np.array(weight_initializer(weights, len(x_train[0]), n_nodes_out)) 
bias = np.array(bias_initializer(bias, n_nodes_out))


weights, bias = training(weights, bias, X_train, Y_train)
