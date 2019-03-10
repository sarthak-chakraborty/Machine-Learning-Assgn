import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

words = []
f = open("./dataset for part 2/words.txt","r")
for x in f:
	words.append(x)

print(len(words))

X_train = [[0]*len(words)]
f = open("./dataset for part 2/traindata.txt","r")
i, j = 0, 1
for x in f:
	doc = int(x.split('\t')[0])
	if(doc != j):
		X_train.append([0]*len(words))
		i += 1
	X_train[i][int(x.split('\t')[1])-1] = 1
	j = doc

print(len(X_train))

Y_train = []
f = open("./dataset for part 2/trainlabel.txt","r")
count = 1
for x in f:
	if(count != 1017):
		Y_train.append(int(x))
	count += 1

print(len(Y_train))


X_test = [[0]*len(words)]
f = open("./dataset for part 2/testdata.txt","r")
i, j = 0, 1
for x in f:
	doc = int(x.split('\t')[0])
	if(doc != j):
		X_test.append([0]*len(words))
		i += 1
	X_test[i][int(x.split('\t')[1])-1] = 1
	j = doc

print(len(X_test))

Y_test = []
f = open("./dataset for part 2/testlabel.txt","r")
for x in f:
	Y_test.append(int(x))
	
print(len(Y_test))


count01 = 0
count02 = 0
count11 = 0
count12 = 0
for i in range(len(X_train)):
	if(X_train[i][484] == 0):
		if(Y_train[i] == 1):
			count01 += 1
		else:
			count02 += 1
	elif(X_train[i][484] == 1):
		if(Y_train[i] == 1):
			count11 += 1
		else:
			count12 += 1

print(count01)
print(count02)
print(count11)
print(count12)


class DecisionTree:
	def __init__(self, criteria, depth):
		self.criteria = criteria
		self.depth = depth
		self.children = []
		self.features = []
		self.measure = []
		self.labels = []
		self.n_nodes = 1
		self.children.append(-1)
		self.features.append(-1)
		self.labels.append(-1)
		self.measure.append(-1)

	def compute_gini(self, X):
		length = len(X)
		n = [0, 0]
		for i in X:
			n[i-1] += 1
		gini = 1 - sum((np.array(n)/(float(length)+np.finfo(float).eps))**2)
		return gini

	def combine_measure(self, measure, ni, n):
		gini = np.array(measure)
		ni = np.array(ni)
		return np.sum((gini*ni)/n)

	def compute_entropy(self, X):
		length = len(X)
		n = [0, 0]
		for i in X:
			n[i-1] += 1

		prob = np.array(n)/(float(length)+np.finfo(float).eps)
		entropy = 0
		for i in prob:
			entropy += i*np.log2(i) if i else 0

		return entropy*(-1)

	def fit(self, X, Y):
		if(self.criteria == 'gini'):
			flag = 1
		elif(self.criteria == 'entropy'):
			flag = 0
		self.fit_DT(X, Y, 0, flag, -1, 0)


	def fit_DT(self, X, Y, node, flag, prev_attr, level):
		if(level > self.depth):
			return

		n_features = len(X[0])
		node_measure = self.compute_gini(Y) if flag else self.compute_entropy(Y)
		self.measure[node] = node_measure

		if(node_measure == 0.0):
			self.labels[node] = max(Y, key=Y.count)
			return

		measure_feature = []
		for i in range(n_features):
			feat = [0,1]
			measure = []
			ni = []
			for l in feat:
				Y_new = []
				for k in range(len(X)):
					if(X[k][i]==l):
						Y_new.append(Y[k])
				measure.append(self.compute_gini(Y_new) if flag else self.compute_entropy(Y_new))
				ni.append(len(Y_new))
			measure_feature.append(self.combine_measure(measure, ni, len(X)))

		measure_feature[prev_attr] = 1 if prev_attr!=-1 else measure_feature[prev_attr]
		attribute = measure_feature.index(min(measure_feature))
		self.features[node] = attribute
		self.measure[node] = node_measure-measure_feature[attribute] if (flag==0) else node_measure
		
		dic = {}
		for l in np.unique(zip(*X)[attribute]):
			self.features.append(-1)
			self.children.append(-1)
			self.labels.append(-1)
			self.measure.append(-1)
			dic[l] = self.n_nodes
			self.n_nodes +=1
		self.children[node] = dic
		self.labels[node] = max(Y, key=Y.count)

		for l in np.unique(zip(*X)[attribute]):
			X_new = []
			Y_new = []
			for i in range(len(X)):
				if(X[i][attribute] == l):
					X_new.append(X[i])
					Y_new.append(Y[i])

			self.fit_DT(X_new, Y_new, self.children[node][l], flag, attribute, level+1)

	def predict(self, X):
		label = []
		for i in X:
			node = 0
			while(1):
				output = self.labels[node]
				next_node = self.children[node][i[self.features[node]]]
				if(self.children[next_node] == -1):
					output = self.labels[next_node]
					break
				else:
					node = next_node
			label.append(output)
		return label

	def accuracy(self, X, Y):
		pred = self.predict(X)
		correct = 0
		for i in range(len(Y)):
			if(Y[i] == pred[i]):
				correct += 1
		return float(correct)/len(Y)


acc_train = []
acc_test = []
for i in range(10,11):
	print(i)
	clf1 = DecisionTree(criteria='entropy',depth=i)
	clf1.fit(X_train, Y_train)
	acc = clf1.accuracy(X_test, Y_test)
	acc_test.append(acc)
	acc = clf1.accuracy(X_train, Y_train)
	acc_train.append(acc)

print(clf1.measure[0])
print(clf1.features[0])
print(clf1.accuracy(X_test, Y_test))


# plt.figure()
# plt.plot([i for i in range(1,30)], acc_train)
# plt.plot([i for i in range(1,30)], acc_test, color='r')
# plt.savefig("Plot1.png")