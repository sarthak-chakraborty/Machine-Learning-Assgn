NAME: Sarthak Chakraborty
ROLL: 16CS30044


###################################
	     README
###################################

This readme file will tell you how to run the code.



Part 1
---------
1. Part1.py performs two types of heirarchical clustering, viz, complete linkage and single linkage.

2. The type of clustering that the user wants to perform can be specified while making an object of the class 'Cluster'. For complete linkage, use the string 'complete', and for single linkage, 'single'.

3. The class contains an attribute cluster() that is a dictionary where the key is the number of clusters and value is a list of list containing the cluster points.

4. The cluster points are saved in a '.npy' file in the same folder.

5. Run the code using `python2 Part1.py`.





Part 2
---------
1. Part2.py performs the Girvan Newman Graph clustering algorithm.

2. Python library 'networkx' must be installed in the system.

3. The class 'Cluster' contains an attribute cluster() that is a dictionary where the key is the number of clusters and value is a list of list containing the cluster points.

4. Input graph is constructed from the data where nodes represent each document and an edge indicates if the two documents has similarity(Jaccard Coefficient) above some threshold. The threshold can be changed by changing the value of the variable 'THRESHOLD'.

5. The input and the output network graphs are saved with the threshold value.

6. The cluster points are saved in a '.npy' file in the same folder.

7. Run the code using `python2 Part2.py`.





Part 3
---------
1. Calculates NMI value of the different clusters formed from various clustering algorithms.

2. Loads the clusters from a '.npy' file. The data must be a list of lists.

3. Gold standard value is chosen from the dataset.

4. Run the code using `python2 Part3.py`.



############################################################################
