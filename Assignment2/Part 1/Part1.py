import numpy as np
import pandas as pd

df1 = pd.read_csv("./dataset for part 1 - Training Data.csv")
df2 = pd.read_csv("./dataset for part 1 - Test Data.csv")

df1 = df1.replace(to_replace=['high','med','low','yes','no'],value=[2,1,0,1,0])
df2 = df2.replace(to_replace=['high','med','low','yes','no'],value=[2,1,0,1,0])

feature = []
for data in df1.iloc[:,:]:
	feature.append(str(data).strip())

X_train = [list(df1.iloc[i,:-1]) for i in range(len(df1))]
Y_train = list(df1[feature[-1]])
X_test = [list(df2.iloc[i,:-1]) for i in range(len(df2))]
Y_test = list(df2[feature[-1]])



class DecisionTree:
	def __init__(self, criteria):
		self.criteria = criteria
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
			n[i] += 1
		gini = 1 - sum((np.array(n)/float(length))**2)
		return gini

	def combine_measure(self, measure, ni, n):
		gini = np.array(measure)
		ni = np.array(ni)
		return np.sum((gini*ni)/n)

	def compute_entropy(self, X):
		length = len(X)
		n = [0, 0]
		for i in X:
			n[i] += 1

		prob = np.array(n)/float(length)
		entropy = 0
		for i in prob:
			entropy += i*np.log2(i) if i else 0

		return entropy*(-1)

	def fit(self, X, Y):
		if(self.criteria == 'gini'):
			flag = 1
		elif(self.criteria == 'entropy'):
			flag = 0
		self.fit_DT(X, Y, 0, flag, -1)


	def fit_DT(self, X, Y, node, flag, prev_attr):
		n_features = len(X[0])
		node_measure = self.compute_gini(Y) if flag else self.compute_entropy(Y)
		self.measure[node] = node_measure

		if(node_measure == 0.0):
			self.labels[node] = max(Y, key=Y.count)
			return

		measure_feature = []
		for i in range(n_features):
			feat = np.unique(zip(*X)[i])
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

			self.fit_DT(X_new, Y_new, self.children[node][l], flag, attribute)

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



def print_DT(children, split_attr, label, feature, node, level):
	if(children[node] != -1):
		print("")
	else:
		if(label[node]==0):
			print(": no")
		else:
			print(": yes")
		return

	for key in children[node]:
		for i in range(level):
			print("\t"),
		print("|"+feature[split_attr[node]]+" ="),
		if(split_attr[node]==0 or split_attr[node]==1):
			if(key==0):
				print("low"),
			elif(key==1):
				print("med"),
			elif(key==2):
				print("high"),
		elif(split_attr[node]==2):
			print(key),
		elif(split_attr[node]==3):
			if(key==0):
				print("no"),
			else:
				print("yes"),
		a = children[node][key]
		print_DT(children, split_attr, label, feature, a, level+1)




clf1 = DecisionTree('gini')
clf1.fit(X_train, Y_train)

print("\n#################################")
print("DECISION TREE trained using Gini Split")
print("#################################")
print("Structure: ")
print_DT(clf1.children, clf1.features, clf1.labels, feature, 0, 0)
print("_______________")
print("\nGini Index of Root Node: " + "%.4f"%clf1.measure[0])
ans = clf1.predict(X_test)
print("\nLabels generated on test data: ")
for i in range(len(ans)):
	if(ans[i]==0):
		print(str(i+1)+ ". no")
	else:
		print(str(i+1)+ ". yes")

acc = clf1.accuracy(X_test, Y_test)
print("\nAccuracy on Test Data: " + str(acc))
print("\n")

print("\n----------------------------------\n")

clf2 = DecisionTree('entropy')
clf2.fit(X_train, Y_train)

print("\n#################################")
print("DECISION TREE trained using Information Gain")
print("#################################")
print("Structure: ")
print_DT(clf2.children, clf2.features, clf2.labels, feature, 0, 0)
print("_______________")
print("\nInformation Gain of Root Node: " + "%.4f"%clf2.measure[0])
ans = clf2.predict(X_test)
print("\nLabels generated on test data: ")
for i in range(len(ans)):
	if(ans[i]==0):
		print(str(i+1)+ ". no")
	else:
		print(str(i+1)+ ". yes")

acc = clf1.accuracy(X_test, Y_test)
print("\nAccuracy on Test Data: " + str(acc))
