import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import graphviz 

df1 = pd.read_csv("./dataset for part 1 - Training Data.csv")
df2 = pd.read_csv("./dataset for part 1 - Test Data.csv")

df1 = df1.replace(to_replace=['high','med','low','yes','no'],value=[2,1,0,1,0])
df2 = df2.replace(to_replace=['high','med','low','yes','no'],value=[2,1,0,1,0])

feature = []
for data in df1.iloc[:,:]:
	feature.append(str(data).strip())

X_train = df1[feature[:-1]]
Y_train = df1[feature[-1]]
X_test = df2[feature[:-1]]
Y_test = df2[feature[-1]]



clf1 = DecisionTreeClassifier(random_state=0).fit(X_train, Y_train)
label_gini = clf1.predict(X_test)
acc_gini = clf1.score(X_test, Y_test)

children_left = clf1.tree_.children_left
children_right = clf1.tree_.children_right
features = clf1.tree_.feature
measure = clf1.tree_.impurity

print("\nDECISION TREE trained using Gini Split")
print("-------------------------------------")
print("Gini Index of Root Node: %.4f" %measure[0])
print("\nLabels generated on test data: ")
for i in range(len(label_gini)):
	if(label_gini[i]==0):
		print(str(i+1)+ ". no")
	else:
		print(str(i+1)+ ". yes")
print("\nAccuracy on Test Data: " + str(acc_gini))

dot_data = tree.export_graphviz(clf1, out_file=None, feature_names=feature[:-1], class_names=['yes','no'])
graph = graphviz.Source(dot_data) 
graph.render("Gini")


print("\n")


clf2 = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(X_train,Y_train)
label_entropy = clf2.predict(X_test)
acc_entropy = clf2.score(X_test, Y_test)

children_left = clf2.tree_.children_left
children_right = clf2.tree_.children_right
features = clf2.tree_.feature
measure = clf2.tree_.impurity
samples = clf2.tree_.n_node_samples

info_gain = measure[0] - (samples[1]*measure[1] + samples[2]*measure[2])/samples[0]

print("\nDECISION TREE trained using Information Gain")
print("-------------------------------------")
print("Information Gain of Root Node: %.4f" %info_gain)
print("\nLabels generated on test data: ")
for i in range(len(label_entropy)):
	if(label_entropy[i]==0):
		print(str(i+1)+ ". no")
	else:
		print(str(i+1)+ ". yes")
print("\nAccuracy on Test Data: " + str(acc_entropy))

dot_data = tree.export_graphviz(clf2, out_file=None, feature_names=feature[:-1], class_names=['yes','no'])
graph = graphviz.Source(dot_data) 
graph.render("Entropy")



# import numpy as np
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

# xls = pd.ExcelFile('dataset for part 1.xlsx')
# df1 = pd.read_excel(xls, sheet_name='Training Data')
# df2 = pd.read_excel(xls, sheet_name='Test Data')

# train_data = []
# test_data = []
# feature = []


# for data in df1.iloc[:,:]:
# 	feature.append(str(data))

# del feature[-1]

# for i in range(len(df1)):
# 	row = []
# 	for data in df1.iloc[i,:]:
# 		row.append(str(data))
# 		# if(type(data) == unicode):
# 		# 	row.append(str(data))
# 		# else:
# 		# 	row.append(data)
# 	train_data.append(row)


# for i in range(len(df2)):
# 	row = []
# 	for data in df2.iloc[i,:]:
# 		row.append(str(data))
# 		# if(type(data) == unicode):
# 		# 	row.append(str(data))
# 		# else:
# 		# 	row.append(data)
# 	test_data.append(row)


# X_train = []
# Y_train = []
# X_test = []
# Y_test = []

# X_train, Y_train = [train_data[i][0:-1] for i in range(len(train_data))], [train_data[i][-1] for i in range(len(train_data))]
# X_test, Y_test = [test_data[i][0:-1] for i in range(len(test_data))], [test_data[i][-1] for i in range(len(test_data))]

# print(X_train)
# print(Y_train)

# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(X_train)
# a = enc.transform(X_train).toarray()
# print(a)
# b = enc.transform(X_test).toarray()


# clf1 = DecisionTreeClassifier(random_state=0).fit(a, Y_train)
# label_gini = clf1.predict(b)
# acc_gini = clf1.score(b, Y_test)
# print(acc_gini)

# clf2 = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(a,Y_train)
# label_entropy = clf2.predict(b)
# acc_entropy = clf1.score(b, Y_test)
# print(acc_entropy)

# print(clf1.decision_path(b))
# print("")
# print(clf2.decision_path(b))

# n_nodes = clf1.tree_.node_count
# children_left = clf1.tree_.children_left
# children_right = clf1.tree_.children_right
# features = clf1.tree_.feature
# threshold = clf1.tree_.threshold

# print(n_nodes)
# print(children_left)
# print(features)
# print(children_right)
# print(threshold)



# import graphviz 
# dot_data = tree.export_graphviz(clf1, out_file=None, class_names=['yes','no'])
# graph = graphviz.Source(dot_data) 

# graph.render("Gini")

# print("\n")

# n_nodes = clf2.tree_.node_count
# children_left = clf2.tree_.children_left
# children_right = clf2.tree_.children_right
# features = clf2.tree_.feature
# threshold = clf2.tree_.threshold

# print(n_nodes)
# print(children_left)
# print(features)
# print(children_right)
# print(threshold)



# dot_data = tree.export_graphviz(clf2, out_file=None, class_names=['yes','no'])
# graph = graphviz.Source(dot_data) 

# graph.render("Entropy")
