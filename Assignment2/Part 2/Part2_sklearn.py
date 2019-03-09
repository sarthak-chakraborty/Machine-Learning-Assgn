import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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
	if(int(x.split('\t')[0]) != j):
		X_train.append([0]*len(words))
		i += 1
	X_train[i][int(x.split('\t')[1])-1] = 1
	j = int(x.split('\t')[0])

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
	if(int(x.split('\t')[0]) != j):
		X_test.append([0]*len(words))
		i += 1
	X_test[i][int(x.split('\t')[1])-1] = 1
	j = int(x.split('\t')[0])

print(len(X_test))

Y_test = []
f = open("./dataset for part 2/testlabel.txt","r")
for x in f:
	Y_test.append(int(x))
	
print(len(Y_test))



# depth = int(input("Enter max depth of the tree: "))

acc_train = []
acc_test = []
for depth in range(1,30):
	clf1 = DecisionTreeClassifier(random_state=0, max_depth=depth).fit(X_train, Y_train)
	label_gini = clf1.predict(X_test)
	acc_gini = clf1.score(X_test, Y_test)
	acc_test.append(acc_gini)
	acc_gini = clf1.score(X_train, Y_train)
	acc_train.append(acc_gini)

# print("%.4f" %acc_gini)
plt.figure()
plt.plot([i for i in range(1,30)], acc_train)
plt.plot([i for i in range(1,30)], acc_test, color='r')
plt.savefig("Plot.png")