import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
		row.append(str(data))
		# if(type(data) == unicode):
		# 	row.append(str(data))
		# else:
		# 	row.append(data)
	train_data.append(row)


for i in range(len(df2)):
	row = []
	for data in df2.iloc[i,:]:
		row.append(str(data))
		# if(type(data) == unicode):
		# 	row.append(str(data))
		# else:
		# 	row.append(data)
	test_data.append(row)


X_train = []
Y_train = []
X_test = []
Y_test = []

X_train, Y_train = [train_data[i][0:-1] for i in range(len(train_data))], [train_data[i][-1] for i in range(len(train_data))]
X_test, Y_test = [test_data[i][0:-1] for i in range(len(test_data))], [test_data[i][-1] for i in range(len(test_data))]

print(X_train)
print(Y_train)

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
a = enc.transform(X_train).toarray()
print(a)
b = enc.transform(X_test).toarray()


clf1 = DecisionTreeClassifier(random_state=0).fit(a, Y_train)
label_gini = clf1.predict(b)
acc_gini = clf1.score(b, Y_test)
print(acc_gini)

clf2 = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(a,Y_train)
label_entropy = clf2.predict(b)
acc_entropy = clf1.score(b, Y_test)
print(acc_entropy)

print(clf1.decision_path(b))
print("")
print(clf2.decision_path(b))

n_nodes = clf1.tree_.node_count
children_left = clf1.tree_.children_left
children_right = clf1.tree_.children_right
features = clf1.tree_.feature
threshold = clf1.tree_.threshold

print(n_nodes)
print(children_left)
print(features)
print(children_right)
print(threshold)



import graphviz 
dot_data = tree.export_graphviz(clf1, out_file=None, class_names=['yes','no'])
graph = graphviz.Source(dot_data) 

graph.render("Gini")

print("\n")

n_nodes = clf2.tree_.node_count
children_left = clf2.tree_.children_left
children_right = clf2.tree_.children_right
features = clf2.tree_.feature
threshold = clf2.tree_.threshold

print(n_nodes)
print(children_left)
print(features)
print(children_right)
print(threshold)



dot_data = tree.export_graphviz(clf2, out_file=None, class_names=['yes','no'])
graph = graphviz.Source(dot_data) 

graph.render("Entropy")