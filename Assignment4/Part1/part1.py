import numpy as np 
import pandas as pd 
import nltk
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize




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
	delim = [' ', '\t' , '\n' , '.' , ',' , ':' , '-']
	for x in d:
		a = re.split('\W+', x)
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

	train = []
	for i in range(len(data)):
		one_hot = []
		for j in range(len(data[i])):
			vec = np.array([0 for k in range(len(tokens))])
			index = tokens.index(data[i][j]) if data[i][j] in tokens else -1
			if index is not -1:
				vec[index] = 1
			one_hot.append(vec)
		train.append(one_hot)

	for i in range(len(train)):
		x = np.random.uniform(0,1)
		if(x<0.8):
			x_train.append(train[i])
			y_train.append(label[i])
		else:
			x_test.append(train[i])
			y_test.append(label[i])
	




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

