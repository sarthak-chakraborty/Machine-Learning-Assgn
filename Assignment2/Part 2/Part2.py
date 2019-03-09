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