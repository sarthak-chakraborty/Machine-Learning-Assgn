import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Get the word features as string
words = []
f = open("./dataset for part 2/words.txt","r")
for x in f:
	words.append(x)


# One-hot-encoding of the training data
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


# Training labels
Y_train = []
f = open("./dataset for part 2/trainlabel.txt","r")
for x in f:
	Y_train.append(int(x))


# One-hot-encoding of the test data
X_test = [[0]*len(words)]
f = open("./dataset for part 2/testdata.txt","r")
i, j = 0, 1
for x in f:
	if(int(x.split('\t')[0]) != j):
		X_test.append([0]*len(words))
		i += 1
	X_test[i][int(x.split('\t')[1])-1] = 1
	j = int(x.split('\t')[0])


# Test labels
Y_test = []
f = open("./dataset for part 2/testlabel.txt","r")
for x in f:
	Y_test.append(int(x))
	


# Get training and test accuracy of the tree for a particular depth taken as input
depth = int(input("Enter max depth of the tree: "))
clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=depth).fit(X_train, Y_train)
acc_train = clf.score(X_train, Y_train)
acc_test = clf.score(X_test, Y_test)
print("Train Accuracy for Decision Tree on depth=" + str(depth) + ": " + str(acc_train))
print("Test Accuracy for Decision Tree on depth=" + str(depth) + ": " + str(acc_test))

print("\n\n##################################")
print("Now we will generate a plot of accuracy vs maximum depth by varying the depth. Calculating for each depth...")


# Get the test and train accuracy for tree with increasing maximum depth
acc_train = []
acc_test = []
prev_train_acc = 0
i = 1
while(prev_train_acc != 1.0):
	clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=i).fit(X_train, Y_train)
	label = clf.predict(X_test)
	acc = clf.score(X_test, Y_test)
	acc_test.append(acc)
	acc = clf.score(X_train, Y_train)
	acc_train.append(acc)
	prev_train_acc = acc
	print("Depth " + str(i) + " done.")
	i += 1

print("\n\n")

# Plot a graph between accuracy vs maximum depth
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

# Find the depth of the tree having the highest test accuracy
highest_depth = np.argmax(acc_test) + 1

plt.axvline(x=highest_depth, color='k', linestyle='--')
plt.savefig("Scikit Learn Implementation.png")

clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=highest_depth).fit(X_train, Y_train)

print("\n\nHighest test accuracy is attained at depth = " + str(highest_depth))
print("Train Accuracy: " + str(clf.score(X_train, Y_train)))
print("Test Accuracy: " + str(clf.score(X_test, Y_test)))