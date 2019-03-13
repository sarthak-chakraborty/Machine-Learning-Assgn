import numpy as np
import matplotlib.pyplot as plt

words = []
f = open("./dataset for part 2/words.txt","r")
for x in f:
	words.append(x.strip())


X_train = [[0]*len(words)]
f = open("./dataset for part 2/traindata.txt","r")
i, j = 0, 1
for x in f:
	doc = int(x.split('\t')[0])
	if(doc==1018 and j==1016):
		X_train.append([0]*len(words))
		i += 1
	if(doc != j):
		X_train.append([0]*len(words))
		i += 1
	X_train[i][int(x.split('\t')[1])-1] = 1
	j = doc



Y_train = []
f = open("./dataset for part 2/trainlabel.txt","r")
for x in f:
	Y_train.append(int(x))


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


Y_test = []
f = open("./dataset for part 2/testlabel.txt","r")
for x in f:
	Y_test.append(int(x))
	




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
			self.labels[node] = max(Y, key=Y.count)
			return

		n_features = len(X[0])
		node_measure = self.compute_gini(Y) if flag else self.compute_entropy(Y)
		self.measure[node] = node_measure

		if(node_measure == 0.0):
			self.labels[node] = max(Y, key=Y.count)
			return

		measure_feature = []
		for i in range(n_features):
			# print(i)
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
		attribute = np.argmin(measure_feature)
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



def print_DT(children, split_attr, label, words, node, level):
	if(children[node] != -1):
		print("")
	else:
		print(": "+str(label[node]))
		return

	for key in children[node]:
		for i in range(level):
			print("   "),
		print("|"+words[split_attr[node]]+" ="),
		print(key),
		a = children[node][key]
		print_DT(children, split_attr, label, words, a, level+1)



depth = int(input("Enter max depth of the tree: "))
clf = DecisionTree(criteria='entropy', depth=depth-1)
clf.fit(X_train, Y_train)
acc_train = clf.accuracy(X_train, Y_train)
acc_test = clf.accuracy(X_test, Y_test)
print("Train Accuracy for Decision Tree on depth=" + str(depth) + ": " + str(acc_train))
print("Test Accuracy for Decision Tree on depth=" + str(depth) + ": " + str(acc_test))


print("\n\n##################################")
print("Now we will generate a plot of accuracy vs maximum depth by varying the depth. Calculating for each depth...")


acc_train = []
acc_test = []
prev_train_acc = 0
i = 1
while(prev_train_acc != 1):
	clf = DecisionTree(criteria='entropy',depth=i-1)
	clf.fit(X_train, Y_train)
	acc = clf.accuracy(X_test, Y_test)
	acc_test.append(acc)
	acc = clf.accuracy(X_train, Y_train)
	prev_train_acc = acc
	acc_train.append(acc)
	print("Depth " + str(i) + " done.")
	i += 1


print("Plotting Train and Test accuracy vs Maximum Depth Curve...")
plt.figure()
plt.plot([j for j in range(1,i)], acc_train)
plt.plot([j for j in range(1,i)], acc_test, color='r')
plt.title("Train, Test Accuracy vs Maximum Depth")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.xticks([2*j for j in range(0,i/2)])
plt.legend(["Train Accuracy","Test Accuracy"])
print("Plotting Done.")

highest_depth = np.argmax(acc_test) + 1

plt.axvline(x=highest_depth, color='k', linestyle='--')
plt.savefig("My Implementation.png")

clf = DecisionTree(criteria='entropy', depth=highest_depth-1)
clf.fit(X_train, Y_train)

print("\n\nHighest test accuracy is attained at depth = " + str(highest_depth))
print("Train Accuracy: " + str(clf.accuracy(X_train, Y_train)))
print("Test Accuracy: " + str(clf.accuracy(X_test, Y_test)))
print("\nTree Structure:")
print("-------------------")
print_DT(clf.children, clf.features, clf.labels, words, 0, 0)