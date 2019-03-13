NAME: Sarthak Chakraborty
ROLL: 16CS30044


###################################
	     README
###################################

This readme file will tell you how to run the code.


Part 1
------------
Part 1 folder has two '.py' files, viz, Part1.py and Part1_sklearn.py. Part1.py contains the my implementation of the decision tree. I have implemented the Decision Tree using classes which can be run by calling an object of the class.

Depth of the decision tree cannot be specified. Only the splitting criteria can be specified by using an attribute 'criteria' while calling the object. 'criteria' must take value 'gini' or 'entropy'.

Data needs to be in the same directory as the codes are present with the corresponding names 'dataset for part 1 - Test Data.csv' and 'dataset for part 1 - Training Data.csv' in csv format.

Data is read using DataFrames in pandas library (if not present install pandas - `pip install pandas`). 

For running 'Part1_sklearn.py', module 'graphviz' must be present which generates a graph for the decision tree built by the sklearn. To install graphviz, type `pip install graphviz`.

For running 'Part1.py', the data must be stored as a list of dimension=(n_samples, n_features) for X_train and X_test. For Y_train and Y_test, the data must be stored in an array of dimension=(n_samples)

Run using `python2 Part1.py` or `python2 Part1_sklearn.py`.




Part 2
------------
Part 2 folder has two '.py' files, viz, Part2.py and Part2_sklearn.py. Part2.py contains the my implementation of the decision tree. I have implemented the Decision Tree using classes which can be run by calling an object of the class.

Depth of the decision tree can be specified by the 'depth' attribute along with 'critera' while calling the object. Value of the attribute is a non-negative integer.

If a tree of depth N is desired, that is, the leaf nodes are at level N, then 'depth' attribute must hold a value of N-1.

Data needs to be in the 'dataset for part 2' folder in the same directory as the codes are present with the corresponding names 'words.txt' having the list of words, 'traindata.txt', 'trainlabel.txt', 'testdata.txt' and 'testlabel.txt'.

Plots are made for both the '.py' files which are saved in the same directory.

For running 'Part2.py', the data must be stored as a list of dimension=(n_samples, n_features) for X_train and X_test. For Y_train and Y_test, the data must be stored in an array of dimension=(n_samples)

Run using `python2 Part2.py` or `python2 Part2_sklearn.py`.



GENERAL
---------------
Read the comments for a better walkthrough of the code. Integer Encoding in used in Part 1 while One-hot-encoding is used in Part 2.


#####################################################################################################################################