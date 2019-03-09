import numpy as np
import pandas as pd

xls = pd.ExcelFile('dataset for part 1.xlsx')
df1 = pd.read_excel(xls, sheet_name='Training Data')
df2 = pd.read_excel(xls, sheet_name='Test Data')

train_data = []
test_data = []
feature = []


for data in df1.iloc[:,:]:
	feature.append(str(data))

del feature[-1]

for i in range(len(df1)):
	row = []
	for data in df1.iloc[i,:]:
		if(type(data) == unicode):
			row.append(str(data))
		else:
			row.append(data)
	train_data.append(row)


for i in range(len(df2)):
	row = []
	for data in df2.iloc[i,:]:
		if(type(data) == unicode):
			row.append(str(data))
		else:
			row.append(data)
	test_data.append(row)


X_train = []
Y_train = []
X_test = []
Y_test = []


for i in range(0, len(train_data)):
	row = []
	for j in range(0,len(train_data[i])):
		if(j==len(train_data[i])-1):
			if(train_data[i][j]=='yes'):
				Y_train.append(1)
			else:
				Y_train.append(0)
		else:
			if(train_data[i][j]=='low'):
				row.append(0)
			elif(train_data[i][j]=='med'):
				row.append(1)
			elif(train_data[i][j]=='high'):
				row.append(2)
			elif(train_data[i][j]=='yes'):
				row.append(1)
			elif(train_data[i][j]=='no'):
				row.append(0)
			else:
				row.append(int(train_data[i][j]))
	X_train.append(row)

for i in range(0, len(test_data)):
	row = []
	for j in range(0,len(test_data[i])):
		if(j==len(test_data[i])-1):
			if(test_data[i][j]=='yes'):
				Y_test.append(1)
			else:
				Y_test.append(0)
		else:
			if(test_data[i][j]=='low'):
				row.append(0)
			elif(test_data[i][j]=='med'):
				row.append(1)
			elif(test_data[i][j]=='high'):
				row.append(2)
			elif(test_data[i][j]=='yes'):
				row.append(1)
			elif(test_data[i][j]=='no'):
				row.append(0)
			else:
				row.append(int(test_data[i][j]))
	X_test.append(row)



class DecisionTree:
	def __init__(self, criteria):
		self.criteria = criteria
		self.children = []
		self.children_right = []
		self.features = []
		self.gini = []
		self.info_gain = []
		self.node = 1

	def compute_gini(self, X):
		length = len(X)
		n = [0, 0]
		for i in X:
			n[i] += 1
		gini = 1 - sum((np.array(n)/float(length))**2)
		return gini

	def combine_gini(self, gini, ni, n):
		gini = np.array(gini)
		ni = np.array(ni)

		return np.sum((gini*ni)/n)


	def fit(self, X, Y):
		n_features = len(X[0])
		if(self.criteria == 'gini'):
			gini = self.compute_gini(Y)
		elif(self.criteria == 'entropy'):
			compute_info_gain(Y)

		if(gini == 0.0):
			self.features.append(-1)
			self.children.append(-1)
			print(self.features)
			print(self.children)
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
				measure.append(self.compute_gini(Y_new))
				ni.append(len(Y_new))
			measure_feature.append(self.combine_gini(measure, ni, len(X)))

		attribute = measure_feature.index(min(measure_feature))
		self.features.append(attribute)
		dic = {}
		for l in np.unique(zip(*X)[attribute]):
			dic[l] = self.node
			self.node +=1
		self.children.append(dic)

		print("Attribute chosen: "+str(attribute))
		print(self.features)
		print(measure_feature)
		print(self.children)

		for l in np.unique(zip(*X)[attribute]):
			X_new = []
			Y_new = []
			for i in range(len(X)):
				if(X[i][attribute] == l):
					X_new.append(X[i])
					Y_new.append(Y[i])
				
			print("X_new" + str(X_new))
			print("Y_new" + str(Y_new))
			print("")
			self.fit(X_new, Y_new)



		# del X[attribute]

			


clf = DecisionTree('gini')
clf.fit(X_train, Y_train)