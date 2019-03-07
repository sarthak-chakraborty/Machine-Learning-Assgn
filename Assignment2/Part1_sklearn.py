import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

xls = pd.ExcelFile('dataset for part 1.xlsx')
df1 = pd.read_excel(xls, sheet_name='Training Data')
df2 = pd.read_excel(xls, sheet_name='Test Data')

train_data = []
test_data = []
feature = []


for data in df1.iloc[:,:]:
	feature.append(str(data))


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

clf1 = DecisionTreeClassifier(random_state=0).fit(X_train, Y_train)
label_gini = clf1.predict(X_test)
acc_gini = clf1.score(X_test, Y_test)

clf2 = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(X_train,Y_train)
label_entropy = clf2.predict(X_test)
acc_entropy = clf1.score(X_test, Y_test)

print(clf1.decision_path(X_test))
print("")
print(clf2.decision_path(X_test))